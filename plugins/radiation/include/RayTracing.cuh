/** \file "RayTracing.cu.h" This file contains definitions and helper functions for CUDA/OptiX routines

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef RAYTRACING_CUH
#define RAYTRACING_CUH

#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef unsigned int uint;

// launch parameters
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(unsigned int, random_seed, , );
rtDeclareVariable(unsigned int, launch_offset, , );
rtDeclareVariable(unsigned int, camera_pixel_offset_x, , );
rtDeclareVariable(unsigned int, camera_pixel_offset_y, , );
rtDeclareVariable(optix::int2, camera_resolution_full, , );
rtDeclareVariable(unsigned int, launch_face, , );
rtDeclareVariable(unsigned int, Nbands_launch, , );
rtDeclareVariable(unsigned int, Nbands_global, , );
rtBuffer<bool, 1> band_launch_flag;

// ray types
rtDeclareVariable(unsigned int, direct_ray_type, , );
rtDeclareVariable(unsigned int, diffuse_ray_type, , );
rtDeclareVariable(unsigned int, camera_ray_type, , );
rtDeclareVariable(unsigned int, pixel_label_ray_type, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim, rtLaunchDim, );

// texture-related buffers
rtBuffer<bool, 3> maskdata;
rtBuffer<int2, 1> masksize;
rtBuffer<int, 1> maskID;
rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;

// Object ID for a primitive
rtBuffer<uint, 1> objectID;
rtBuffer<int2, 1> object_subdivisions;

// ID for first primitive in an object
rtBuffer<uint, 1> primitiveID;

// UUID to array position lookup (indexed by UUID, returns array position)
// Size: max_UUID + 1 (sparse array)
// Usage: uint position = primitive_positions[UUID];
rtBuffer<uint, 1> primitive_positions;

// Radiation sources buffers
rtDeclareVariable(unsigned int, Nsources, , );
rtBuffer<float, 1> source_fluxes;
rtBuffer<float, 1> source_fluxes_cam; // Camera-weighted source fluxes for specular [source * Nbands * Ncameras]
rtBuffer<float3, 1> source_positions;
rtBuffer<float3, 1> source_rotations;
rtBuffer<float2, 1> source_widths;
rtBuffer<unsigned int, 1> source_types;

// Diffuse radiation
rtBuffer<float, 1> diffuse_flux;
rtBuffer<float, 1> diffuse_extinction;
rtBuffer<float3, 1> diffuse_peak_dir;
rtBuffer<float, 1> diffuse_dist_norm;
rtBuffer<float, 1> Rsky;

// Atmospheric sky radiance parameters (for camera miss rays)
rtBuffer<float4, 1> sky_radiance_params; // Per-band: (circumsolar_strength, circumsolar_width, horizon_coeff, zenith_scale)
rtBuffer<float, 1> camera_sky_radiance; // Per-band: base sky radiance (W/m²/sr) for camera atmospheric model
rtDeclareVariable(float3, sun_direction, , ); // Sun direction vector for sky radiance evaluation

// Solar disk rendering for camera (lens flare support)
rtBuffer<float, 1> solar_disk_radiance; // Per-band: solar disk radiance (W/m²/sr) when looking directly at sun
rtDeclareVariable(float, solar_disk_cos_angle, , ); // Cosine of solar angular radius (~0.265°, cos ≈ 0.99999)

//--- Patches ---//
rtBuffer<float3, 2> patch_vertices;
rtBuffer<unsigned int, 1> patch_UUID;
//--- Triangles ---//
rtBuffer<float3, 2> triangle_vertices;
rtBuffer<unsigned int, 1> triangle_UUID;
//--- Disks ---//
// To-Do: disks not finished
rtBuffer<unsigned int, 1> disk_UUID;
//--- Tiles ---//
rtBuffer<float3, 2> tile_vertices;
rtBuffer<unsigned int, 1> tile_UUID;
//--- Voxels ---//
rtBuffer<float3, 2> voxel_vertices;
rtBuffer<unsigned int, 1> voxel_UUID;
//--- Bounding Box ---//
rtBuffer<float3, 2> bbox_vertices;
rtBuffer<unsigned int, 1> bbox_UUID;

// Primitive data
rtDeclareVariable(unsigned int, Nprimitives, , );
rtDeclareVariable(unsigned int, bbox_UUID_base, , ); // Starting UUID for bboxes (max_real_UUID + 1)
rtDeclareVariable(float2, periodic_flag, , );
rtBuffer<char, 1> twosided_flag;
rtBuffer<float, 2> transform_matrix;
rtBuffer<unsigned int, 1> primitive_type;
rtBuffer<float, 1> primitive_solid_fraction;

rtBuffer<float, 1> rho, tau;
rtBuffer<float, 1> rho_cam, tau_cam;

rtBuffer<float, 1> specular_exponent;
rtBuffer<float, 1> specular_scale;
rtDeclareVariable(unsigned int, specular_reflection_enabled, , );
rtDeclareVariable(unsigned int, scattering_iteration, , ); // Current scattering iteration (0-based)

// Output buffers
rtBuffer<float, 1> radiation_in;
rtBuffer<float, 1> radiation_in_camera;
rtBuffer<float, 1> radiation_out_top;
rtBuffer<float, 1> radiation_out_bottom;
rtBuffer<float, 1> scatter_buff_top;
rtBuffer<float, 1> scatter_buff_bottom;
rtBuffer<float, 1> scatter_buff_top_cam;
rtBuffer<float, 1> scatter_buff_bottom_cam;
rtBuffer<float, 1> radiation_specular; // Incident radiation for specular (per source, camera-weighted) [source * Ncameras * Nprimitives * Nbands + camera * Nprimitives * Nbands + primitive * Nbands + band]

// Camera variables
rtBuffer<unsigned int, 1> camera_pixel_label;
rtBuffer<float, 1> camera_pixel_depth;
rtDeclareVariable(unsigned int, camera_ID, , );
rtDeclareVariable(unsigned int, Ncameras, , );
rtDeclareVariable(float3, camera_position, , );
rtDeclareVariable(float2, camera_direction, , );
rtDeclareVariable(float, camera_lens_diameter, , );
rtDeclareVariable(float, FOV_aspect_ratio, , );
rtDeclareVariable(float, camera_focal_length, , ); // Focal plane distance (working distance for ray generation), NOT lens optical focal length
rtDeclareVariable(float, camera_viewplane_length, , );
rtDeclareVariable(float, camera_pixel_solid_angle, , ); // Solid angle subtended by a single pixel (steradians)


void queryGPUMemory(void);

/** OptiX ray payload - contains data for each ray traversal. */
struct PerRayData {

    //! "strength" or amount of energy associated with the ray.
    double strength;
    //! UUID of primitive from which ray originated
    uint origin_UUID;
    //! Face of primitive from which ray was launched (true=top, false=bottom)
    bool face;
    //! Seed for curand random number generator
    uint seed;
    //! Number of periodic boundary intersections for ray
    // unsigned char periodic_depth;
    //! Numerical identifier for radiation source corresponding to each ray
    /**
     * \note The data type limits to a maximum of 256 radiation sources
     */
    unsigned char source_ID;
    //! Flag to determine if ray hit a periodic boundary
    bool hit_periodic_boundary;
    optix::float3 periodic_hit;
};

static __device__ void init_state(PerRayData *prd);

__device__ __inline__ void atomicFloatAdd(float *address, float val) {
#if __CUDA_ARCH__ >= 200
    atomicAdd(address, val);
#else
    float old = val;
    while ((old = atomicExch(address, atomicExch(address, 0.0f) + old)) != 0.0f)
        ;
#endif
} // end atomic float add

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ __inline__ double atomicAdd(double *address, double val) {
    unsigned long long int *address_as_ull = (unsigned long long int *) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// -- random number generation -- //

template<unsigned int N>
static __host__ __device__ __inline__ unsigned int tea(unsigned int val0, unsigned int val1) {
    unsigned int v0 = val0;
    unsigned int v1 = val1;
    unsigned int s0 = 0;

    for (unsigned int n = 0; n < N; n++) {
        s0 += 0x9e3779b9;
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    return v0;
}

static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

// Generate random float in [0, 1)
static __host__ __device__ __inline__ float rnd(unsigned int &prev) {
    return ((float) lcg(prev) / (float) 0x01000000);
}

/** Function to rotate a 3D vector given spherical angles elevation and azimuth ON THE GPU. */
__device__ float3 d_rotatePoint(const float3 &position, const float &theta, const float &phi) {

    /** @param[in] "position" 3D coordinate of point to be rotated. */
    /** @param[in] "theta" elevation angle of rotation. */
    /** @param[in] "phi" azimuthal angle of rotation. */

    float Ry[3][3], Rz[3][3];

    float st = sinf(theta);
    float ct = cosf(theta);

    float sp = sinf(phi);
    float cp = cosf(phi);

    // Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
    Ry[0][0] = ct;
    Ry[0][1] = 0.f;
    Ry[0][2] = st;
    Ry[1][0] = 0.f;
    Ry[1][1] = 1.f;
    Ry[1][2] = 0.f;
    Ry[2][0] = -st;
    Ry[2][1] = 0.f;
    Ry[2][2] = ct;

    Rz[0][0] = cp;
    Rz[0][1] = -sp;
    Rz[0][2] = 0.f;
    Rz[1][0] = sp;
    Rz[1][1] = cp;
    Rz[1][2] = 0.f;
    Rz[2][0] = 0.f;
    Rz[2][1] = 0.f;
    Rz[2][2] = 1.f;

    // Multiply Ry*Rz

    float rotMat[3][3] = {0.f};

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                rotMat[i][j] = rotMat[i][j] + Rz[i][k] * Ry[k][j];
            }
        }
    }

    // Multiply the rotation matrix with the position vector.
    float3 tmp;
    tmp.x = rotMat[0][0] * position.x + rotMat[0][1] * position.y + rotMat[0][2] * position.z;
    tmp.y = rotMat[1][0] * position.x + rotMat[1][1] * position.y + rotMat[1][2] * position.z;
    tmp.z = rotMat[2][0] * position.x + rotMat[2][1] * position.y + rotMat[2][2] * position.z;

    return tmp;
}

__device__ void d_makeTransformMatrix(optix::float3 rotation, float (&T)[16]) {

    T[0] = cosf(rotation.z) * cosf(rotation.y);
    T[1] = cosf(rotation.z) * sinf(rotation.y) * sinf(rotation.x) - sinf(rotation.z) * cosf(rotation.x);
    T[2] = cosf(rotation.z) * sinf(rotation.y) * cosf(rotation.x) + sinf(rotation.z) * sinf(rotation.x);
    T[3] = 0.f;
    T[4] = sinf(rotation.z) * cosf(rotation.y);
    T[5] = sinf(rotation.z) * sinf(rotation.y) * sinf(rotation.x) + cosf(rotation.z) * cosf(rotation.x);
    T[6] = sinf(rotation.z) * sinf(rotation.y) * cosf(rotation.x) - cosf(rotation.z) * sinf(rotation.x);
    T[7] = 0.f;
    T[8] = -sinf(rotation.y);
    T[9] = cosf(rotation.y) * sinf(rotation.x);
    T[10] = cosf(rotation.y) * cosf(rotation.x);
    T[11] = 0.f;
    T[12] = 0.f;
    T[13] = 0.f;
    T[14] = 0.f;
    T[15] = 1.f;
}

__device__ void d_invertMatrix(const float *m, float *minv) {

    float inv[16], det;

    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];

    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];

    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];

    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];

    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];

    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];

    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];

    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];

    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];

    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];

    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];

    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];

    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];

    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];

    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];

    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];

    det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];

    // if (det == 0)
    // return false;

    det = 1.0 / det;

    for (int i = 0; i < 16; i++)
        minv[i] = inv[i] * det;
}


//! Function to transform a 3D point based on current affine transformation matrix on the GPU
__device__ void d_transformPoint(const float (&transform_matrix)[16], float3 &v) {

    float3 V;

    V.x = transform_matrix[0] * v.x + transform_matrix[1] * v.y + transform_matrix[2] * v.z + transform_matrix[3];
    V.y = transform_matrix[4] * v.x + transform_matrix[5] * v.y + transform_matrix[6] * v.z + transform_matrix[7];
    V.z = transform_matrix[8] * v.x + transform_matrix[9] * v.y + transform_matrix[10] * v.z + transform_matrix[11];

    v = V;
}

//! Function to calculate the magnitude of float3 vector on the GPU
__device__ float d_magnitude(const float3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

//! Function to calculate the surface area of a primitive
__device__ float d_calculatePrimitiveArea(const uint primitive_type, const float (&transform_matrix)[16]) {

    if (primitive_type == 0 || primitive_type == 3) { // Patch or AlphaMask

        float3 s0 = make_float3(0, 0, 0);
        float3 s1 = make_float3(1, 1, 0);
        float3 s2 = make_float3(0, 1, 0);
        d_transformPoint(transform_matrix, s0);
        d_transformPoint(transform_matrix, s1);
        d_transformPoint(transform_matrix, s2);
        return d_magnitude(s2 - s0) * d_magnitude(s2 - s1);

    } else if (primitive_type == 1) { // Triangle

        float3 v0 = make_float3(0, 0, 0);
        d_transformPoint(transform_matrix, v0);
        float3 v1 = make_float3(0, 1, 0);
        d_transformPoint(transform_matrix, v1);
        float3 v2 = make_float3(1, 1, 0);
        d_transformPoint(transform_matrix, v2);
        float3 A(v1 - v0);
        float3 B(v2 - v0);
        float3 C(v2 - v1);
        float a = d_magnitude(A);
        float b = d_magnitude(B);
        float c = d_magnitude(C);
        float s = 0.5f * (a + b + c);
        return sqrtf(s * (s - a) * (s - b) * (s - c));

    } else if (primitive_type == 2) { // Disk

        float r0 = transform_matrix[0];
        float r1 = transform_matrix[5];
        return M_PI * r0 * r1;
    }

    return 0;
}

__device__ void d_matMult33(float *ml, float *mr, float *m) {

    for (int i = 0; i < 9; i++) {
        m[i] = 0.f;
    }

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                m[i * 3 + j] = m[i * 3 + j] + ml[i * 3 + k] * mr[k * 3 + j];
            }
        }
    }
}

__device__ void d_matMult44(float *ml, float *mr, float *m) {

    for (int i = 0; i < 16; i++) {
        m[i] = 0.f;
    }

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            for (int k = 0; k < 4; k++) {
                m[i * 4 + j] = m[i * 4 + j] + ml[i * 4 + k] * mr[k * 4 + j];
            }
        }
    }
}

__device__ void d_sampleDisk(uint &seed, optix::float3 &sample) {

    // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

    optix::float3 sp;

    float Rx = rnd(seed);
    float Ry = rnd(seed);

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
    sp.x = r * cos(p);
    sp.y = r * sin(p);
    sp.z = 0.f;

    sample = sp;
}

__device__ void d_sampleSquare(uint &seed, optix::float3 &sample) {

    optix::float3 sp;

    float Rx = rnd(seed);
    float Ry = rnd(seed);

    sp.x = -0.5f + Rx;
    sp.y = -0.5f + Ry;
    sp.z = 0.f;

    sample = sp;
}

__device__ bool d_sampleTexture_patch(optix::float3 &sp, const optix::int2 &subpatch_index, const optix::float2 &subpatch_size, PerRayData &prd, int mask_ID, int uv_ID) {

    int2 sz = masksize[mask_ID];
    uint3 ind;

    bool solid = false;
    int count = 0;
    while (!solid) {
        count++;

        float2 uv = make_float2((sp.x + 0.5f), (sp.y + 0.5f));
        if (uv_ID == -1) { // does not have custom (u,v) coordinates
            ind = make_uint3(floorf(float(sz.x - 1) * uv.x), floorf(float(sz.y - 1) * (1.f - uv.y)), mask_ID);
        } else { // has custom (u,v) coordinates
            float2 uvmin = uvdata[make_uint2(0, uv_ID)];
            float2 duv;
            duv.x = uvdata[make_uint2(1, uv_ID)].x - uvdata[make_uint2(0, uv_ID)].x;
            duv.y = uvdata[make_uint2(2, uv_ID)].y - uvdata[make_uint2(1, uv_ID)].y;
            ind = make_uint3(floorf(float(sz.x - 1) * (uvmin.x + uv.x * duv.x)), floorf(float(sz.y - 1) * (1.f - uvmin.y - uv.y * duv.y)), mask_ID);
        }
        solid = maskdata[ind];
        if (!solid) {
            if (count > 10) {
                break;
            }
            sp.x = -0.5f + (subpatch_index.x + rnd(prd.seed)) * subpatch_size.x;
            sp.y = -0.5f + (subpatch_index.y + rnd(prd.seed)) * subpatch_size.y;
        }
    }

    return solid;
}

__device__ bool d_sampleTexture_triangle(optix::float3 &sp, const optix::float3 &v0, const optix::float3 &v1, const optix::float3 &v2, PerRayData &prd, const float (&m_trans)[16], int mask_ID, int uv_ID) {

    int2 sz = masksize[mask_ID];

    float2 uv0 = uvdata[make_uint2(0, uv_ID)];
    float2 uv1 = uvdata[make_uint2(1, uv_ID)];
    float2 uv2 = uvdata[make_uint2(2, uv_ID)];

    float a = v0.x - v1.x, b = v0.x - v2.x, d = v0.x;
    float e = v0.y - v1.y, f = v0.y - v2.y, h = v0.y;
    float i = v0.z - v1.z, j = v0.z - v2.z, l = v0.z;

    bool solid = false;
    int count = 0;
    while (!solid) {
        count++;

        float3 R = sp;
        d_transformPoint(m_trans, R);

        float c = R.x, g = R.y, k = R.z;

        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
        float q = g * i - e * k, s = e * j - f * i;

        float inv_denom = 1.f / (a * m + b * q + c * s);

        float e1 = d * m - b * n - c * p;
        float beta = e1 * inv_denom;

        float r = r = e * l - h * i;
        float e2 = a * n + d * q + c * r;
        float gamma = e2 * inv_denom;

        float2 uv = uv0 + beta * (uv1 - uv0) + gamma * (uv2 - uv0);

        uint3 ind = make_uint3(floorf(float(sz.x - 1) * uv.x), floorf(float(sz.y - 1) * (1.f - uv.y)), mask_ID);

        if (uv.x < 0 || uv.x > 1.f || uv.y < 0 || uv.y > 1.f) {
            solid = true;
        } else {
            solid = maskdata[ind];
        }
        if (!solid) {
            if (count > 10) {
                break;
            }
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);
            if (Rx < Ry) {
                sp.x = Rx;
                sp.y = Ry;
            } else {
                sp.x = Ry;
                sp.y = Rx;
            }
        }
    }

    return solid;
}

static __host__ __device__ __inline__ float acos_safe(float x) {
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return acos(x);
}

static __host__ __device__ __inline__ float asin_safe(float x) {
    if (x < -1.0)
        x = -1.0;
    else if (x > 1.0)
        x = 1.0;
    return asin(x);
}

// ----- auxiliary helpers --------------------------------------
static __forceinline__ __device__ uint32_t reverseBits32(uint32_t v) {
    // bitwise reverse (Van-der-Corput direction numbers for dim0)
    v = (v << 16) | (v >> 16);
    v = ((v & 0x00ff00ffU) << 8) | ((v & 0xff00ff00U) >> 8);
    v = ((v & 0x0f0f0f0fU) << 4) | ((v & 0xf0f0f0f0U) >> 4);
    v = ((v & 0x33333333U) << 2) | ((v & 0xccccccccU) >> 2);
    v = ((v & 0x55555555U) << 1) | ((v & 0xaaaaaaaaU) >> 1);
    return v;
}

// Simple PCG-style 32-bit hash (acts as an Owen scramble)
static __forceinline__ __device__ uint32_t pcgHash(uint32_t v) {
    v ^= v >> 17;
    v *= 0xed5ad4bbU;
    v ^= v >> 11;
    v *= 0xac4c1b51U;
    v ^= v >> 15;
    v *= 0x31848babU;
    v ^= v >> 14;
    return v;
}

// Convert uint32→float in [0,1)
static __forceinline__ __device__ float uint32ToUnitFloat(uint32_t v) {
    return v * 2.3283064365386963e-10f; // 1 / 2^32
}

// Sobol dimension-0 (Van der Corput) with Owen scramble
static __forceinline__ __device__ float sobolDim0(uint32_t idx, uint32_t scramble) {
    return uint32ToUnitFloat(reverseBits32(idx) ^ scramble);
}

// Sobol dimension-1 (direction numbers from poly x^2+x+1)
static __forceinline__ __device__ float sobolDim1(uint32_t idx, uint32_t scramble) {
    uint32_t v = 1u << 31; // MSB first
    uint32_t res = scramble;
    while (idx) {
        if (idx & 1)
            res ^= v;
        idx >>= 1;
        v ^= v >> 1; // Gray-code update
    }
    return uint32ToUnitFloat(res);
}

// ----- public entry point -------------------------------------
static __forceinline__ __device__ float2 sobol2D(uint32_t sampleIdx) {
    // Independent Owen scrambles for each dimension
    uint32_t scramble0 = pcgHash(sampleIdx);
    uint32_t scramble1 = pcgHash(sampleIdx ^ 0x9e3779b9U); // golden-ratio XOR

    float x = sobolDim0(sampleIdx, scramble0); // dimension 0
    float y = sobolDim1(sampleIdx, scramble1); // dimension 1
    return make_float2(x, y);
}

// simple LCG → xorshift → float conversion (1 cycle, 4 ops)
__forceinline__ __device__ float uint32_to_unit_float(uint32_t v) {
    return v * 2.3283064365386963e-10f; // 1 / 2^32
}
__forceinline__ __device__ float2 hash2D(uint32_t s) {
    s ^= s * 0x6c50b47cu;
    s ^= s >> 17;
    float u = uint32_to_unit_float(s);

    s ^= s * 0xb82f1e52u;
    s ^= s >> 17;
    float v = uint32_to_unit_float(s);

    return make_float2(u, v);
}

// Ray-source intersection helpers for camera rendering

//! Test if ray intersects a spherical radiation source
/**
 * \param[in] ray_origin Origin point of the ray
 * \param[in] ray_direction Direction vector of the ray (must be normalized)
 * \param[in] sphere_center Center position of the sphere
 * \param[in] sphere_radius Radius of the sphere
 * \return True if ray intersects sphere in front of origin
 */
__device__ __forceinline__ bool d_raySphereIntersect(const optix::float3 &ray_origin, const optix::float3 &ray_direction, const optix::float3 &sphere_center, float sphere_radius) {
    optix::float3 oc = ray_origin - sphere_center;
    float b = optix::dot(oc, ray_direction);
    float c = optix::dot(oc, oc) - sphere_radius * sphere_radius;
    float discriminant = b * b - c;

    if (discriminant < 0.0f)
        return false;

    float t = -b - sqrtf(discriminant);
    return t > 0.0f; // Hit if in front of camera
}

//! Test if ray intersects a rectangular radiation source
/**
 * \param[in] ray_origin Origin point of the ray
 * \param[in] ray_direction Direction vector of the ray (must be normalized)
 * \param[in] rect_center Center position of the rectangle
 * \param[in] rect_width Width of the rectangle (x dimension)
 * \param[in] rect_length Length of the rectangle (y dimension)
 * \param[in] rect_rotation Rotation angles (rx, ry, rz) of the rectangle
 * \param[out] out_cos_angle Cosine of angle between ray and rectangle normal
 * \return True if ray intersects front face of rectangle in front of origin
 */
__device__ __forceinline__ bool d_rayRectangleIntersect(const optix::float3 &ray_origin, const optix::float3 &ray_direction, const optix::float3 &rect_center, float rect_width, float rect_length, const optix::float3 &rect_rotation,
                                                        float &out_cos_angle) {
    // Build transform matrix
    float transform[16];
    d_makeTransformMatrix(rect_rotation, transform);

    // Get normal vector (transformed +Z axis)
    optix::float3 normal = optix::make_float3(transform[2], transform[6], transform[10]);

    // Ray-plane intersection
    float denom = optix::dot(ray_direction, normal);
    if (denom >= -1e-6f)
        return false; // Parallel or back-facing

    optix::float3 oc = rect_center - ray_origin;
    float t = optix::dot(oc, normal) / denom;
    if (t <= 0.0f)
        return false; // Behind camera

    // Hit point in world space
    optix::float3 hit_point = ray_origin + t * ray_direction;

    // Transform to local coordinates
    optix::float3 local_hit = hit_point - rect_center;
    float inv_transform[16];
    d_invertMatrix(transform, inv_transform);
    d_transformPoint(inv_transform, local_hit);

    // Point-in-rectangle test
    if (fabsf(local_hit.x) > rect_width * 0.5f)
        return false;
    if (fabsf(local_hit.y) > rect_length * 0.5f)
        return false;

    out_cos_angle = -denom; // Cosine of angle between ray and normal
    return true;
}

//! Test if ray intersects a disk radiation source
/**
 * \param[in] ray_origin Origin point of the ray
 * \param[in] ray_direction Direction vector of the ray (must be normalized)
 * \param[in] disk_center Center position of the disk
 * \param[in] disk_radius Radius of the disk
 * \param[in] disk_rotation Rotation angles (rx, ry, rz) of the disk
 * \param[out] out_cos_angle Cosine of angle between ray and disk normal
 * \return True if ray intersects front face of disk in front of origin
 */
__device__ __forceinline__ bool d_rayDiskIntersect(const optix::float3 &ray_origin, const optix::float3 &ray_direction, const optix::float3 &disk_center, float disk_radius, const optix::float3 &disk_rotation, float &out_cos_angle) {
    // Build transform matrix
    float transform[16];
    d_makeTransformMatrix(disk_rotation, transform);

    // Get normal vector (transformed +Z axis)
    optix::float3 normal = optix::make_float3(transform[2], transform[6], transform[10]);

    // Ray-plane intersection
    float denom = optix::dot(ray_direction, normal);
    if (denom >= -1e-6f)
        return false; // Parallel or back-facing

    optix::float3 oc = disk_center - ray_origin;
    float t = optix::dot(oc, normal) / denom;
    if (t <= 0.0f)
        return false; // Behind camera

    // Hit point in world space
    optix::float3 hit_point = ray_origin + t * ray_direction;

    // Point-in-disk test (distance from center)
    optix::float3 offset = hit_point - disk_center;
    float dist_sq = optix::dot(offset, offset);
    if (dist_sq > disk_radius * disk_radius)
        return false;

    out_cos_angle = -denom; // Cosine of angle between ray and normal
    return true;
}


#endif
