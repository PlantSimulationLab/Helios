/** \file "rayHit.cu" Functions for object intersection.

    Copyright (C) 2016-2025 Brian Bailey

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

using namespace optix;

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable( unsigned int, UUID, attribute UUID, );

RT_PROGRAM void closest_hit_direct(){

    uint objID = objectID[UUID];

    if( (periodic_flag.x==1 || periodic_flag.y==1) && primitive_type[objID] == 5 ){ //periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x  = xbounds.y - xbounds.x;
        float width_y  = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.x)<=eps) {//-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.y)<=eps) {//+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.x)<=eps ) {//-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.y)<=eps) {//+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    }

};

RT_PROGRAM void closest_hit_diffuse() {

    uint origin_UUID = prd.origin_UUID;

    uint objID = objectID[UUID];

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[objID] == 5) { //periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x  = xbounds.y - xbounds.x;
        float width_y  = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.x)<=eps) {//-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.y)<=eps) {//+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.x)<=eps ) {//-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.y)<=eps) {//+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    } else {

        //Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being recieved)

        //find out if we hit top or bottom surface
        float3 normal;

        float m[16];
        for (uint i = 0; i < 16; i++) {
            m[i] = transform_matrix[optix::make_uint2(i, objID)];
        }

        if (primitive_type[objID] == 0 || primitive_type[objID] == 3) {//hit patch or tile
            float3 s0 = make_float3(0, 0, 0);
            float3 s1 = make_float3(1, 0, 0);
            float3 s2 = make_float3(0, 1, 0);
            d_transformPoint(m, s0);
            d_transformPoint(m, s1);
            d_transformPoint(m, s2);
            normal = cross(s1 - s0, s2 - s0);
        } else if (primitive_type[UUID] == 1) {//hit triangle
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(0, 1, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(1, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v1);
        } else if (primitive_type[UUID] == 2) {//hit disk
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(1, 0, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(0, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[UUID] == 4) {//hit voxel
            float3 vmin = make_float3(-0.5, -0.5, -0.5);
            d_transformPoint(m, vmin);
            float3 vmax = make_float3(0.5, 0.5, 0.5);
            d_transformPoint(m, vmax);
        }
        normal = normalize(normal);

        bool face = dot(normal, ray.direction) < 0;

        int b = -1;
        for (int b_global = 0; b_global < Nbands_global; b_global++) {

            if( band_launch_flag[b_global]==0 ){
                continue;
            }
            b++;

            size_t ind_origin = Nbands_launch * origin_UUID + b;
            size_t ind_hit = Nbands_launch * UUID + b;

            double strength;
            if (face || primitive_type[objID] == 4) {
                strength = radiation_out_top[ ind_hit ] * prd.strength;
            } else {
                strength = radiation_out_bottom[ ind_hit ] * prd.strength;
            }

            if( strength==0 ){
                continue;
            }

            size_t radprop_ind_global = Nprimitives*Nbands_global*prd.source_ID + Nbands_global*origin_UUID + b_global;
            float t_rho = rho[ radprop_ind_global ];
            float t_tau = tau[ radprop_ind_global ];

            if (primitive_type[objectID[origin_UUID]] == 4) { //ray was launched from voxel

//                float kappa = t_rho; //just a reminder that rho is actually the absorption coefficient
//                float sigma_s = t_tau; //just a reminder that tau is actually the scattering coefficient
//                float beta = kappa + sigma_s;
//
//                // absorption
//                atomicAdd(&radiation_in[ind_origin], strength * exp(-beta * 0.5 * prd.area) * kappa / beta);
//
//                // scattering
//                atomicAdd(&scatter_buff_top[ind_origin], strength * exp(-beta * 0.5 * prd.area) * sigma_s / beta);

            } else { //ray was NOT launched from voxel

                // absorption
                atomicAdd(&radiation_in[ind_origin], strength * (1.f - t_rho - t_tau));

                if ((t_rho > 0 || t_tau > 0) && strength > 0) {
                    if (prd.face) {//reflection from top, transmission from bottom
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_rho); //reflection
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_tau); //transmission
                    } else {//reflection from bottom, transmission from top
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_rho); //reflection
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_tau); //transmission
                    }
                }
                if( Ncameras>0 ) {
                    size_t indc = prd.source_ID*Nprimitives*Nbands_global*Ncameras + origin_UUID*Nbands_global*Ncameras + b_global*Ncameras + camera_ID;
                    float t_rho_cam = rho_cam[ indc ];
                    float t_tau_cam = tau_cam[ indc ];
                    if ((t_rho_cam > 0 || t_tau_cam > 0) && strength > 0) {
                        if (prd.face) {//reflection from top, transmission from bottom
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_rho_cam); //reflection
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam); //transmission
                        } else {//reflection from bottom, transmission from top
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam); //reflection
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_tau_cam); //transmission
                        }
                    }
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

    uint objID = objectID[UUID];

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[objID] == 5) { //periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x  = xbounds.y - xbounds.x;
        float width_y  = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.x)<=eps) {//-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.y)<=eps) {//+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.x)<=eps ) {//-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.y)<=eps) {//+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    } else {

        //Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being received)

        //find out if we hit top or bottom surface // per each ray
        float3 normal;

        float m[16];
        for (uint i = 0; i < 16; i++) {
            m[i] = transform_matrix[optix::make_uint2(i, objID)];
        }

        if (primitive_type[objID] == 0 || primitive_type[objID] == 3) {//hit patch or tile
            float3 s0 = make_float3(0, 0, 0);
            float3 s1 = make_float3(1, 0, 0);
            float3 s2 = make_float3(0, 1, 0);
            d_transformPoint(m, s0);
            d_transformPoint(m, s1);
            d_transformPoint(m, s2);
            normal = cross(s1 - s0, s2 - s0);
        } else if (primitive_type[UUID] == 1) {//hit triangle
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(0, 1, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(1, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v1);
        } else if (primitive_type[UUID] == 2) {//hit disk
            float3 v0 = make_float3(0, 0, 0);
            d_transformPoint(m, v0);
            float3 v1 = make_float3(1, 0, 0);
            d_transformPoint(m, v1);
            float3 v2 = make_float3(0, 1, 0);
            d_transformPoint(m, v2);
            normal = cross(v1 - v0, v2 - v0);
        } else if (primitive_type[UUID] == 4) {//hit voxel
            float3 vmin = make_float3(-0.5, -0.5, -0.5);
            d_transformPoint(m, vmin);
            float3 vmax = make_float3(0.5, 0.5, 0.5);
            d_transformPoint(m, vmax);
        }
        normal = normalize(normal);

        bool face = dot(normal, ray.direction) < 0;


        float3 camera_normal = d_rotatePoint( make_float3(0,0,1), -0.5*M_PI+camera_direction.x, 0.5f*M_PI-camera_direction.y );

        double strength;
        for( size_t b=0; b<Nbands_launch; b++ ) {

            if (face || primitive_type[objID] == 4) {
                strength = radiation_out_top[Nbands_launch * UUID + b] * prd.strength;   //this one  /fabs(dot())
            } else {
                strength = radiation_out_bottom[Nbands_launch * UUID + b] * prd.strength;
            }


            //specular reflection

//            float strength_spec = 0;
//            for( int rr=0; rr<Nsources; rr++ ) {
//
//                //light direction
//                float3 light_direction;
//                float light_magnitude;
//                float spec = 0;
//                if (source_types[rr] == 0) { //collimated source
//                    light_direction = normalize(source_positions[rr]);
//                    spec = fabs(dot(normal, light_direction)) * source_fluxes[rr*Nbands_launch + b];
//
//                } else { //sphere source
//
//                    float3 ray_origin = ray.origin + t_hit * ray.direction;
//
//                    //sample point on surface of sphere
//                    float theta_s = acos_safe(1.f - 2.f * rnd(prd.seed));
//                    float phi_s = rnd(prd.seed) * 2.f * M_PI;
//                    float3 sphere_point = 0.5 * source_widths[rr].x * make_float3(sin(theta_s) * cos(phi_s), sin(theta_s) * sin(phi_s), cos(theta_s));
//
//                    light_direction = sphere_point + source_positions[rr] - ray_origin;
//
//                    light_magnitude = d_magnitude(light_direction);
//                    light_direction = normalize(light_direction);
//                    spec = fabs(dot(normal, light_direction)) * source_fluxes[ rr*Nbands_launch + b] * fabs(dot(normal, light_direction)) / (light_magnitude * light_magnitude)*source_widths[rr].x*source_widths[rr].x;
//                }
//
//                float3 specular_direction = normalize(2 * abs(dot(light_direction, normal)) * normal - light_direction);
//
//                strength_spec += spec * powf( abs(dot(specular_direction,ray.direction)), 1.7f) *1.f/float(launch_dim.x*launch_dim.y);
//
////                printf("strength_spec = %f %f %f %f\n",spec,specular_direction.x,specular_direction.y,specular_direction.z);
//
//            }

            // absorption

            atomicAdd(&radiation_in_camera[Nbands_launch * prd.origin_UUID + b], strength );
//            atomicAdd(&radiation_in_camera[Nbands_launch * prd.origin_UUID + b], strength+strength_spec );

        }

    }
}

RT_PROGRAM void closest_hit_pixel_label() {

    uint origin_UUID = prd.origin_UUID;

    uint objID = objectID[UUID];

    if ((periodic_flag.x == 1 || periodic_flag.y == 1) && primitive_type[objID] == 5) { //periodic boundary condition

        prd.hit_periodic_boundary = true;

        float3 ray_origin = ray.origin + t_hit * ray.direction;

        float eps = 1e-5;

        float2 xbounds = make_float2(bbox_vertices[make_uint2(0, 0)].x, bbox_vertices[make_uint2(1, 1)].x);
        float2 ybounds = make_float2(bbox_vertices[make_uint2(0, 0)].y, bbox_vertices[make_uint2(1, 1)].y);

        float width_x  = xbounds.y - xbounds.x;
        float width_y  = ybounds.y - ybounds.x;

        prd.periodic_hit = ray_origin;
        if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.x)<=eps) {//-x facing boundary
            prd.periodic_hit.x += +width_x - eps;
        } else if (periodic_flag.x == 1 && fabs(ray_origin.x-xbounds.y)<=eps) {//+x facing boundary
            prd.periodic_hit.x += -width_x + eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.x)<=eps ) {//-y facing boundary
            prd.periodic_hit.y += +width_y - eps;
        } else if (periodic_flag.y == 1 && fabs(ray_origin.y-ybounds.y)<=eps) {//+y facing boundary
            prd.periodic_hit.y += -width_y + eps;
        }

    } else {

        //Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being received)
        //Note: We are reserving a value of 0 for the sky, so we will store UUID+1
        camera_pixel_label[origin_UUID] = UUID+1;

        float depth = prd.strength + t_hit;
        float3 camera_direction3 = d_rotatePoint( make_float3(1,0,0), -0.5*M_PI+camera_direction.x, 0.5f*M_PI-camera_direction.y );
        camera_pixel_depth[origin_UUID] = abs(dot(camera_direction3, ray.direction))*depth;

    }
}

RT_PROGRAM void miss_direct(){

    uint objID = objectID[prd.origin_UUID];

    int b=-1;
    for( int b_global=0; b_global<Nbands_global; b_global++ ) {

        if( band_launch_flag[b_global]==0 ){
            continue;
        }
        b++;

        size_t ind_origin = Nbands_launch*prd.origin_UUID+b;

        size_t radprop_ind_global = Nprimitives*Nbands_global*prd.source_ID + Nbands_global*prd.origin_UUID + b_global;
        float t_rho = rho[ radprop_ind_global ];
        float t_tau = tau[ radprop_ind_global ];

        double strength = prd.strength * source_fluxes[ prd.source_ID*Nbands_launch + b ];

        //absorption
        atomicAdd(&radiation_in[ind_origin], strength * (1.f - t_rho - t_tau)  );

        if (t_rho > 0 || t_tau > 0) {
            if (prd.face) {//reflection from top, transmission from bottom
                atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_rho); //reflection
                atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_tau); //transmission
            } else {//reflection from bottom, transmission from top
                atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_rho); //reflection
                atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_tau); //transmission
            }
        }
        if( Ncameras>0 ) {
            size_t indc = prd.source_ID*Nprimitives*Nbands_global*Ncameras + prd.origin_UUID*Nbands_global*Ncameras + b_global*Ncameras + camera_ID;
            float t_rho_cam = rho_cam[ indc ];
            float t_tau_cam = tau_cam[ indc ];
            if ( (t_rho_cam > 0 || t_tau_cam > 0) && strength>0 ) {
                if (prd.face) {//reflection from top, transmission from bottom
                    atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_rho_cam); //reflection
                    atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam); //transmission
                } else {//reflection from bottom, transmission from top
                    atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam); //reflection
                    atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_tau_cam); //transmission
                }
            }
        }

    }

}

RT_PROGRAM void miss_diffuse() {

//    double strength;
//    if (prd.face || primitive_type[objectID[prd.origin_UUID]] == 3) {
//        strength = radiation_out_top[prd.origin_UUID] * prd.strength * prd.area;
//    } else {
//        strength = radiation_out_bottom[prd.origin_UUID] * prd.strength * prd.area;
//    }
//
//    atomicFloatAdd(&Rsky[prd.origin_UUID], strength);

    int b=-1;
    for( size_t b_global=0; b_global<Nbands_global; b_global++ ) {

        if( band_launch_flag[b_global]==0 ){
            continue;
        }
        b++;

        if (diffuse_flux[b] > 0.f) {

            size_t ind_origin = Nbands_launch * prd.origin_UUID + b;

            size_t radprop_ind_global = Nprimitives*Nbands_global*prd.source_ID + Nbands_global*prd.origin_UUID + b_global;
            float t_rho = rho[ radprop_ind_global ];
            float t_tau = tau[ radprop_ind_global ];

            if (primitive_type[objectID[prd.origin_UUID]] == 4) { //ray was launched from voxel

                float kappa = t_rho; //just a reminder that rho is actually the absorption coefficient
                float sigma_s = t_tau; //just a reminder that tau is actually the scattering coefficient
                float beta = kappa + sigma_s;

                // absorption
                atomicAdd(&radiation_in[ind_origin], diffuse_flux[b] * prd.strength * kappa / beta);

                // scattering
                atomicAdd(&scatter_buff_top[ind_origin], diffuse_flux[b] * prd.strength * sigma_s / beta);

            } else { //ray was NOT launched from voxel

                float fd = 1.f;
                if ( diffuse_extinction[b] > 0.f) {
                    float psi = acos_safe(dot(diffuse_peak_dir[b], ray.direction));
                    if (psi < M_PI / 180.f) {
                        fd = powf(M_PI / 180.f, -diffuse_extinction[b]) * diffuse_dist_norm[b];   //Replace 'pow' by 'powf' in
                    } else {
                        fd = powf(psi, -diffuse_extinction[b]) * diffuse_dist_norm[b];
                    }
                }

                float strength = fd * diffuse_flux[b] * prd.strength;

                //  absorption
                atomicAdd(&radiation_in[ind_origin], strength * (1.f - t_rho - t_tau) );

                if (t_rho > 0 || t_tau > 0) {
                    if (prd.face) {//reflection from top, transmission from bottom
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_rho); //reflection
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_tau); //transmission
                    } else {//reflection from bottom, transmission from top
                        atomicFloatAdd(&scatter_buff_bottom[ind_origin], strength * t_rho); //reflection
                        atomicFloatAdd(&scatter_buff_top[ind_origin], strength * t_tau); //transmission
                    }
                }
                if( Ncameras>0 ) {
                    size_t indc = prd.source_ID*Nprimitives*Nbands_global*Ncameras + prd.origin_UUID*Nbands_global*Ncameras + b_global*Ncameras + camera_ID;
                    float t_rho_cam = rho_cam[ indc ];
                    float t_tau_cam = tau_cam[ indc ];
                    if ( (t_rho_cam > 0 || t_tau_cam > 0) && prd.strength>0 ) {
                        if (prd.face) {//reflection from top, transmission from bottom
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_rho_cam); //reflection
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_tau_cam); //transmission
                        } else {//reflection from bottom, transmission from top
                            atomicFloatAdd(&scatter_buff_bottom_cam[ind_origin], strength * t_rho_cam); //reflection
                            atomicFloatAdd(&scatter_buff_top_cam[ind_origin], strength * t_tau_cam); //transmission
                        }
                    }
                }
            }

        }


    }

}

RT_PROGRAM void miss_camera() {

    for( size_t b=0; b<Nbands_launch; b++ ) {

        if (diffuse_flux[b] > 0.f) {

            float fd = 1.f;
            if (diffuse_extinction[b] > 0.f) {
                float psi = acos_safe(dot(diffuse_peak_dir[b], ray.direction));
                if (psi < M_PI / 180.f) {
                    fd = powf(float(M_PI) / 180.f, -diffuse_extinction[b]) * diffuse_dist_norm[b];
                } else {
                    fd = powf(psi, -diffuse_extinction[b]) * diffuse_dist_norm[b];
                }
            }

            //absorption
            atomicAdd(&radiation_in_camera[Nbands_launch*prd.origin_UUID+b], fd * diffuse_flux[b] * prd.strength);

        }

    }

}

RT_PROGRAM void miss_pixel_label() {

    camera_pixel_depth[prd.origin_UUID] = -1;

}
