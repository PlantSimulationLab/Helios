/** \file "rayGeneration.cu" File containing OptiX ray generation programs

    Copyright (C) 2016-2023  Brian Bailey

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

rtDeclareVariable(rtObject,      top_object, , );
rtDeclareVariable(unsigned int,  direct_ray_type, , );
rtDeclareVariable(unsigned int,  diffuse_ray_type, , );
rtDeclareVariable(unsigned int,  camera_ray_type, , );
rtDeclareVariable(unsigned int,  pixel_label_ray_type, , );
rtDeclareVariable(unsigned int,  random_seed, , );
rtDeclareVariable(unsigned int,  launch_offset, , );

rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );

//texture-related buffers
rtBuffer<bool, 3>   maskdata;
rtBuffer<int2, 1>   masksize;
rtBuffer<int, 1>   maskID;
rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;

//Buffer data for each primitive
rtBuffer<float, 2>  transform_matrix;
rtBuffer<unsigned int, 1> primitive_type;
rtBuffer<float, 1> primitive_area;
rtBuffer<bool, 1> twosided_flag;
rtBuffer<float, 1>   radiation_out_top;
rtBuffer<float, 1>   radiation_out_bottom;

//Object ID for a primitive
rtBuffer<uint,1> objectID;
rtBuffer<int2, 1> object_subdivisions;

//ID for first primitive in an object
rtBuffer<uint,1> primitiveID;

//Radiation sources buffers
rtDeclareVariable( unsigned int, Nsources, , );
rtBuffer<float, 1> source_fluxes;
rtBuffer<float3, 1> source_positions;
rtBuffer<float3, 1> source_rotations;
rtBuffer<float2, 1> source_widths;
rtBuffer<unsigned int, 1> source_types;


//Camera variables
rtDeclareVariable( float3, camera_position, , );
rtDeclareVariable( float2, camera_direction, , );
rtDeclareVariable( float, camera_lens_diameter, , );
rtDeclareVariable( float2, sensor_size, , );
rtDeclareVariable( float, camera_focal_length, , );
rtDeclareVariable( float, camera_viewplane_length, , );

RT_PROGRAM void direct_raygen()
{

    uint Nrays = launch_dim.x*launch_dim.y;
    uint ray_index = launch_dim.x*launch_index.y + launch_index.x;

    PerRayData prd;
    prd.seed = tea<16>(ray_index+Nrays*launch_index.z,random_seed);

    uint objID = launch_offset+launch_index.z;

    uint pID = primitiveID[objID];
    uint ptype = primitive_type[objID];
    int puvID = uvID[objID];

    float3 sp;

    float3 normal;

    //transformation matrix
    float m_trans[16];
    for( uint i=0; i<16; i++ ){
        m_trans[i] = transform_matrix[ optix::make_uint2(i,objID) ];
    }

    //looping over sub-patches
    int NX = object_subdivisions[objID].x;
    int NY = object_subdivisions[objID].y;
    for( int jj=0; jj<NY; jj++ ){
        for( int ii=0; ii<NX; ii++ ){

            uint UUID = pID + jj*NX + ii;

            //two random samples [0,1]
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

            if( ptype == 0 || ptype == 3 ){ //Patch or Tile

                uint Nx = launch_dim.x;
                uint Ny = launch_dim.y;
                float dx = 1.f/float(NX);
                float dy = 1.f/float(NY);

                // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + ii*dx + float(launch_index.x)*dx/float(Nx) + Rx*dx/float(Nx);
                sp.y = -0.5f + jj*dy + float(launch_index.y)*dy/float(Ny) + Ry*dy/float(Ny);
                sp.z = 0.f;

                int ID = maskID[objID];
                if( ID>=0 && primitive_area[UUID]>0 ){//has texture transparency

                    int2 sz = masksize[ID];
                    uint3 ind;

                    bool solid = false;
                    int count=0;
                    while( !solid ){
                        count ++;

                        float2 uv = make_float2( sp.x+0.5f, 1.f-sp.y-0.5f );
                        if( puvID==-1 ){//does not have custom (u,v) coordinates
                            ind = make_uint3( roundf(float(sz.x-1)*uv.x), roundf(float(sz.y-1)*uv.y), ID );
                        }else{//has custom (u,v) coordinates
                            float2 uvmin = uvdata[ make_uint2(3,puvID) ];
                            float2 duv;
                            duv.x = uvdata[ make_uint2(1,puvID) ].x - uvdata[ make_uint2(0,puvID) ].x;
                            duv.y = uvdata[ make_uint2(1,puvID) ].y - uvdata[ make_uint2(2,puvID) ].y;
                            ind = make_uint3( roundf(float(sz.x-1)*(uvmin.x+uv.x*duv.x)), roundf(float(sz.y-1)*(uvmin.y+uv.y*duv.y)), ID );
                        }
                        solid = maskdata[ind];
                        if( !solid ){
                            if( count>10 ){
                                break;
                            }
                            sp.x = -0.5f + ii*dx + rnd(prd.seed)*dx;
                            sp.y = -0.5f + jj*dy + rnd(prd.seed)*dy;
                        }
                    }

                }

                //calculate rectangle normal vector (world coordinates)
                float3 v0 = make_float3(0,0,0);
                d_transformPoint(m_trans,v0);
                float3 v1 = make_float3(1,0,0);
                d_transformPoint(m_trans,v1);
                float3 v2 = make_float3(0,1,0);
                d_transformPoint(m_trans,v2);

                normal = normalize(cross(v1-v0,v2-v0));

            }else if( ptype == 1 ){ //Triangle

                // Map sample to triangle with vertices (0,0,0), (0,1,0), (1,1,0)
                if( Rx<Ry ){
                    sp.x = Rx;
                    sp.y = Ry;
                }else{
                    sp.x = Ry;
                    sp.y = Rx;
                }
                sp.z = 0;

                //calculate triangle normal vector (world coordinates)
                float3 v0 = make_float3(0,0,0);
                d_transformPoint(m_trans,v0);
                float3 v1 = make_float3(0,1,0);
                d_transformPoint(m_trans,v1);
                float3 v2 = make_float3(1,1,0);
                d_transformPoint(m_trans,v2);

                normal = normalize(cross(v1-v0,v2-v1));

                int ID = maskID[objID];
                if( ID>=0 && primitive_area[UUID]>0 ){//has texture transparency

                    int2 sz = masksize[ID];

                    float2 uv0 = uvdata[ make_uint2(0,puvID) ];
                    float2 uv1 = uvdata[ make_uint2(1,puvID) ];
                    float2 uv2 = uvdata[ make_uint2(2,puvID) ];

                    float a = v0.x - v1.x, b = v0.x - v2.x, d = v0.x;
                    float e = v0.y - v1.y, f = v0.y - v2.y,  h = v0.y;
                    float i = v0.z - v1.z, j = v0.z - v2.z, l = v0.z;

                    bool solid = false;
                    int count=0;
                    while( !solid ){
                        count ++;

                        float3 R = sp;
                        d_transformPoint(m_trans,R);

                        float c = R.x, g = R.y, k = R.z;

                        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
                        float q = g * i - e * k, s = e * j - f * i;

                        float inv_denom  = 1.f / (a * m + b * q + c * s);

                        float e1 = d * m - b * n - c * p;
                        float beta = e1 * inv_denom;

                        float r = r = e * l - h * i;
                        float e2 = a * n + d * q + c * r;
                        float gamma = e2 * inv_denom;

                        float2 uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0);

                        uint3 ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), ID );

                        if( uv.x<0 || uv.x>1.f || uv.y<0 || uv.y>1.f ) {
                            solid = true;
                        }else {
                            solid = maskdata[ind];
                        }
                        if( !solid ){
                            if( count>10 ){
                                break;
                            }
                            Rx = rnd(prd.seed);
                            Ry = rnd(prd.seed);
                            if( Rx<Ry ){
                                sp.x = Rx;
                                sp.y = Ry;
                            }else{
                                sp.x = Ry;
                                sp.y = Rx;
                            }
                        }

                    }
                }

            }else if( ptype == 2 ){ //Disk

                d_sampleDisk( prd.seed, sp );

                //calculate disk normal vector (world coordinates)
                float3 v0 = make_float3(0,0,0);
                d_transformPoint(m_trans,v0);
                float3 v1 = make_float3(1,0,0);
                d_transformPoint(m_trans,v1);
                float3 v2 = make_float3(0,1,0);
                d_transformPoint(m_trans,v2);

                normal = normalize(cross(v1-v0,v2-v0));

            }else if( ptype == 4 ){ //Voxel

                float Rz = rnd(prd.seed);

            }

            //translate the ray to the location of the primitive

            float3 ray_origin = sp;
            d_transformPoint(m_trans,ray_origin);

            //Send a ray toward each source
            for( int rr=0; rr<Nsources; rr++ ){

                //set the ray direction
                float3 ray_direction;
                float ray_magnitude;
                if( source_types[rr]==0 ){ //collimated source
                    ray_direction = normalize(source_positions[rr]);
                    ray_magnitude = RT_DEFAULT_MAX;
//                    prd.strength = source_fluxes[rr]/float(launch_dim.x*launch_dim.y)*fabs(dot(normal,ray_direction));
                    prd.strength = 1./double(launch_dim.x*launch_dim.y)*fabs(dot(normal,ray_direction));
                }else if( source_types[rr]==1 || source_types[rr]==2 ){ //sphere source

                    //sample point on surface of sphere
                    float theta_s = acos_safe(1.f-2.f*rnd(prd.seed));
                    float phi_s = rnd(prd.seed)*2.f*M_PI;
                    float3 sphere_point = 0.5*source_widths[rr].x*make_float3( sin(theta_s)*cos(phi_s), sin(theta_s)*sin(phi_s), cos(theta_s) );

                    ray_direction = sphere_point+source_positions[rr]-ray_origin;

                    ray_magnitude = d_magnitude(ray_direction);
                    ray_direction = normalize( ray_direction );
                    prd.strength = 0.f;
                    uint N = 10;
                    for( uint j=0; j<N; j++ ){
                        for( uint i=0; i<N; i++ ){
                            float theta = acos_safe(1.f-2.f*(float(i)+0.5f)/float(N));
                            float phi = (float(j)+0.5f)*2.f*M_PI/float(N);
                            float3 light_direction = make_float3( sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta) );
                            if( dot(light_direction,ray_direction)<0 ){
//                                prd.strength+=source_fluxes[rr]/float(launch_dim.x*launch_dim.y)*fabs(dot(normal,ray_direction))*fabs(dot(light_direction,ray_direction))/(ray_magnitude*ray_magnitude)/(N*N)*source_widths[rr]*source_widths[rr];
                                prd.strength+=1./double(launch_dim.x*launch_dim.y)*fabs(dot(normal,ray_direction))*fabs(dot(light_direction,ray_direction))/(ray_magnitude*ray_magnitude)/(N*N)*source_widths[rr].x*source_widths[rr].x;
                            }
                        }
                    }

                }else if( source_types[rr]==3 ){ //rectangle source

                    //transformation matrix
                    float light_transform[16];
                    d_makeTransformMatrix(source_rotations[rr],light_transform);

                    //sample point on surface of disk
                    float3 square_point;
                    d_sampleSquare( prd.seed, square_point );
                    square_point = make_float3(source_widths[rr].x*square_point.x,source_widths[rr].y*square_point.y,square_point.z);
                    d_transformPoint( light_transform, square_point );

                    float3 light_direction = make_float3(0,0,1);
                    d_transformPoint( light_transform, light_direction );

                    ray_direction = square_point+source_positions[rr]-ray_origin;

                    if( dot(ray_direction,light_direction)>0.f ){ //don't emit from back side of light source (note that ray goes toward the source, so the dot produce is negative when light is pointed at primitive)
                        continue;
                    }

                    ray_magnitude = d_magnitude(ray_direction);
                    ray_direction = normalize( ray_direction );
                    prd.strength = 1./double(launch_dim.x*launch_dim.y)*fabs(dot(normal,ray_direction))*fabs(dot(light_direction,ray_direction))/(ray_magnitude*ray_magnitude)*source_widths[rr].x*source_widths[rr].y/M_PI;

                }else if( source_types[rr]==4 ){ //disk source

                    //transformation matrix
                    float light_transform[16];
                    d_makeTransformMatrix(source_rotations[rr],light_transform);

                    //sample point on surface of disk
                    float3 disk_point;
                    d_sampleDisk( prd.seed, disk_point );
                    d_transformPoint( light_transform, disk_point );

                    float3 light_direction = make_float3(0,0,1);
                    d_transformPoint( light_transform, light_direction );

                    ray_direction = source_widths[rr].x*disk_point+source_positions[rr]-ray_origin;

                    if( dot(ray_direction,light_direction)>0.f ){ //don't emit from back side of light source (note that ray goes toward the source, so the dot produce is negative when light is pointed at primitive)
                        continue;
                    }

                    ray_magnitude = d_magnitude(ray_direction);
                    ray_direction = normalize( ray_direction );
                    prd.strength = 1./double(launch_dim.x*launch_dim.y)*fabs(dot(normal,ray_direction))*fabs(dot(light_direction,ray_direction))/(ray_magnitude*ray_magnitude)*source_widths[rr].x*source_widths[rr].x;

                }

                optix::Ray ray = optix::make_Ray(ray_origin, ray_direction, direct_ray_type, 1e-4, ray_magnitude);

                prd.origin_UUID = UUID;
                prd.periodic_depth = 0;
                prd.source_ID = rr;

                if( dot( ray_direction, normal )>0 ){
                    prd.face = 1;
                }else{
                    prd.face = 0;
                }

                rtTrace( top_object, ray, prd);

            }

        }
    }

}

RT_PROGRAM void diffuse_raygen(){

    uint dimx = launch_dim.x*launch_dim.y;
    uint indx = launch_dim.x*launch_index.y + launch_index.x;

    PerRayData prd;
    prd.seed = tea<16>(indx+dimx*launch_index.z,random_seed);

    uint objID = launch_offset+launch_index.z;

    uint pID = primitiveID[objID];
    uint ptype = primitive_type[objID];
    int puvID = uvID[objID];

    //transformation matrix
    float m_trans[16];
    for( uint i=0; i<16; i++ ){
        m_trans[i] = transform_matrix[ optix::make_uint2(i,objID) ];
    }

    float3 sp, normal;
    float area;

    //looping over sub-patches
    int NX = object_subdivisions[objID].x;
    int NY = object_subdivisions[objID].y;
    for( int jj=0; jj<NY; jj++ ){
        for( int ii=0; ii<NX; ii++ ){

            uint UUID = pID + jj*NX + ii;

            //two random samples [0,1]
            float Rx = rnd(prd.seed);
            float Ry = rnd(prd.seed);

            if( ptype>4 ){
                printf("objID = %d\n",objID);
                printf("Invalid primitive type in diffuse ray launch.\n");
            }

            if( ptype == 0 || ptype == 3 ){ //Patch or Tile

                //calculate rectangle normal vector (world coordinates)
                float3 s0 = make_float3(0,0,0);
                float3 s1 = make_float3(1,0,0);
                float3 s2 = make_float3(0,1,0);
                d_transformPoint(m_trans,s0);
                d_transformPoint(m_trans,s1);
                d_transformPoint(m_trans,s2);
                area = primitive_area[UUID];

                normal = normalize(cross(s1-s0,s2-s0));

                // Map sample to rectangle [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + (ii+Rx)/float(NX);
                sp.y = -0.5f + (jj+Ry)/float(NY);
                sp.z = 0.f;

                int ID = maskID[objID];
                if( ID>=0 ){//has texture transparency

                    int2 sz = masksize[ID];
                    uint3 ind;

                    bool solid = false;
                    int count=0;
                    while( !solid ){
                        count++;

                        float2 uv = make_float2( sp.x+0.5f, 1.f-sp.y-0.5f );
                        if( puvID==-1 ){//does not have custom (u,v) coordinates
                            ind = make_uint3( roundf(float(sz.x-1)*uv.x), roundf(float(sz.y-1)*uv.y), ID );
                        }else{//has custom (u,v) coordinates
                            float2 uvmin = uvdata[ make_uint2(3,puvID) ];
                            float2 duv;
                            duv.x = uvdata[ make_uint2(1,puvID) ].x - uvdata[ make_uint2(0,puvID) ].x;
                            duv.y = uvdata[ make_uint2(1,puvID) ].y - uvdata[ make_uint2(2,puvID) ].y;
                            ind = make_uint3( roundf(float(sz.x-1)*(uvmin.x+uv.x*duv.x)), roundf(float(sz.y-1)*(uvmin.y+uv.y*duv.y)), ID );
                        }
                        solid = maskdata[ind];
                        if( !solid ){
                            if( count>10 ){
                                break;
                            }
                            sp.x = -0.5f + (ii+rnd(prd.seed))/float(NX);
                            sp.y = -0.5f + (jj+rnd(prd.seed))/float(NY);
                        }
                    }

                }

            }else if( ptype == 1 ){ //Triangle

                // Map sample to triangle with vertices (0,0,0), (0,1,0), (1,1,0)
                if( Rx<Ry ){
                    sp.x = Rx;
                    sp.y = Ry;
                }else{
                    sp.x = Ry;
                    sp.y = Rx;
                }
                sp.z = 0;

                //triangle area
                area = primitive_area[UUID];

                //calculate triangle normal vector (world coordinates)
                float3 v0 = make_float3(0,0,0);
                d_transformPoint(m_trans,v0);
                float3 v1 = make_float3(0,1,0);
                d_transformPoint(m_trans,v1);
                float3 v2 = make_float3(1,1,0);
                d_transformPoint(m_trans,v2);

                normal = normalize(cross(v1-v0,v2-v1));

                int ID = maskID[objID];
                if( ID>=0 ){//has texture transparency

                    int2 sz = masksize[ID];

                    float2 uv0 = uvdata[ make_uint2(0,puvID) ];
                    float2 uv1 = uvdata[ make_uint2(1,puvID) ];
                    float2 uv2 = uvdata[ make_uint2(2,puvID) ];

                    float a = v0.x - v1.x, b = v0.x - v2.x, d = v0.x;
                    float e = v0.y - v1.y, f = v0.y - v2.y,  h = v0.y;
                    float i = v0.z - v1.z, j = v0.z - v2.z, l = v0.z;

                    bool solid = false;
                    int count=0;
                    while( !solid ){
                        count++;

                        float3 R = sp;
                        d_transformPoint(m_trans,R);

                        float c = R.x, g = R.y, k = R.z;

                        float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
                        float q = g * i - e * k, s = e * j - f * i;

                        float inv_denom  = 1.f / (a * m + b * q + c * s);

                        float e1 = d * m - b * n - c * p;
                        float beta = e1 * inv_denom;

                        float r = r = e * l - h * i;
                        float e2 = a * n + d * q + c * r;
                        float gamma = e2 * inv_denom;

                        float2 uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0);

                        uint3 ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), ID );

                        if( uv.x<0 || uv.x>1.f || uv.y<0 || uv.y>1.f ) {
                            solid = true;
                        }else {
                            solid = maskdata[ind];
                        }
                        if( !solid ){
                            if( count>10 ){
                                break;
                            }
                            Rx = rnd(prd.seed);
                            Ry = rnd(prd.seed);
                            if( Rx<Ry ){
                                sp.x = Rx;
                                sp.y = Ry;
                            }else{
                                sp.x = Ry;
                                sp.y = Rx;
                            }
                        }

                    }
                }

            }else if( ptype == 2 ){ //Disk

                // Map Sample to disk - from Suffern (2007) "Ray tracing fom the ground up" Chap. 6

                //first map sample point to rectangle [-1,1] [-1,1]
                sp.x = -1.f + 2.f*Rx;
                sp.y = -1.f + 2.f*Ry;

                float r, p;
                if( sp.x>-sp.y) {
                    if( sp.x > sp.y ){
                        r = sp.x;
                        p = sp.y/sp.x;
                    }else{
                        r = sp.y;
                        p = 2.f-sp.x/sp.y;
                    }
                }else{
                    if( sp.x < sp.y ){
                        r = -sp.x;
                        p = 4.f + sp.y/sp.x;
                    }else{
                        r = -sp.y;
                        if( sp.y!=0.f ){ //avoid division by zero at origin
                            p = 6.f - sp.x/sp.y;
                        }else{
                            p = 0.f;
                        }
                    }
                }
                p*=0.25f*M_PI;

                //find x,y point on unit disk
                sp.x = r*cosf(p);
                sp.y = r*sinf(p);
                sp.z = 0.f;

                area = primitive_area[UUID];

                //calculate disk normal vector (world coordinates)
                float3 v0 = make_float3(0,0,0);
                d_transformPoint(m_trans,v0);
                float3 v1 = make_float3(1,0,0);
                d_transformPoint(m_trans,v1);
                float3 v2 = make_float3(0,1,0);
                d_transformPoint(m_trans,v2);
                normal = normalize(cross(v1-v0,v2-v0));

            }else if( ptype == 4 ){ //Voxel

                // Map sample to cube [-0.5,0.5] [-0.5,0.5] [-0.5,0.5]
                sp.x = -0.5f + Rx;
                sp.y = -0.5f + Ry;
                sp.z = -0.5f + rnd(prd.seed);

            }

            //Choose random hemispherical direction - map samples to hemisphere (from Suffern (2007) "Ray tracing fom the ground up" Chap. 6)

            float Rt;
            float Rp;

            Rt = (launch_index.x + rnd(prd.seed))/float(launch_dim.x);
            Rp = (launch_index.y + rnd(prd.seed))/float(launch_dim.y);

            float t;
            if( ptype==4 ){ //voxel
                t = acos_safe(1.f-Rt);
            }else{ //other
                t = asin_safe(sqrtf(Rt));
            }
            float p = 2.f*M_PI*Rp;

            float3 ray_direction;
            ray_direction.x = sin(t)*cos(p);
            ray_direction.y = sin(t)*sin(p);
            ray_direction.z = cos(t);

            float3 ray_origin;
            optix::Ray ray;

            if( ptype==4 ){ //voxel

                prd.strength = 0.5f/float(dimx);
                prd.area = 0.1;
                prd.origin_UUID = UUID;
                prd.face = 0;
                prd.periodic_depth = 0;
                prd.source_ID = 0;

                ray_origin = sp;
                d_transformPoint(m_trans,ray_origin);

                ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);
                rtTrace( top_object, ray, prd);

                ray = optix::make_Ray(ray_origin, -ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);
                rtTrace( top_object, ray, prd);

            }else{ //not a voxel

                ray_direction = d_rotatePoint( ray_direction, acos_safe(normal.z), atan2(normal.y,normal.x) );

                prd.strength = 1.f/float(dimx);

                prd.area = area;
                prd.origin_UUID = UUID;
                prd.periodic_depth = 0;
                prd.source_ID = 0;

                // ---- "top" surface launch -------
                ray_origin = sp;
                d_transformPoint(m_trans,ray_origin);
                ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);

                prd.face = 1;

                rtTrace( top_object, ray, prd);

                // ---- "bottom" surface launch -------
                if( twosided_flag[objID] ){
                    ray_direction = -ray_direction;
                    ray = optix::make_Ray(ray_origin, ray_direction, diffuse_ray_type, 1e-5, RT_DEFAULT_MAX);

                    prd.face = 0;

                    rtTrace( top_object, ray, prd);

                }

            }

        }
    }

}

RT_PROGRAM void camera_raygen(){

    uint dimx = launch_dim.x*launch_dim.y;  // x number of ray, y width, z length
    uint indx = launch_dim.x*launch_index.y + launch_index.x;

    optix::int2 camera_resolution = optix::make_int2( launch_dim.y, launch_dim.z );

    PerRayData prd;
    prd.seed = tea<16>(indx+dimx*launch_index.z,random_seed);

    float3 sp;

    uint ii = launch_index.y; //x-pixel
    uint jj = launch_index.z; //y-pixel
    size_t origin_ID = jj*launch_dim.y + ii; //global pixel index


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

    float PPointsRatiox =1.f;
    float PPointsRatioy =1.f;
    float Rx = rnd(prd.seed);
    float Ry = rnd(prd.seed);
    //    float Rx = 0.5;
    //    float Ry = 0.5;

    // Map sample to pixel
    sp.y = (-0.5f * PPointsRatioy + (ii+Rx)/float(camera_resolution.x))*sensor_size.x;
    sp.z = (0.5f * PPointsRatiox - (jj+Ry)/float(camera_resolution.y))*sensor_size.y;
    sp.x = camera_viewplane_length;

    // *** Determine point 'p' on focal plane that passes through the lens center (0,0) and pixel sample (view direction coordinate aligned) *** //

    float3 p = make_float3(camera_focal_length,sp.y/camera_viewplane_length*camera_focal_length,sp.z/camera_viewplane_length*camera_focal_length);

    // *** Sample point on lens (view direction coordinate aligned *** //

    float3 ray_origin = make_float3(0,0,0);
    if( camera_lens_diameter>0 ) {
        float3 disk_sample;
        d_sampleDisk(prd.seed, disk_sample);
        ray_origin = make_float3(0.f,0.5f*disk_sample.x*camera_lens_diameter,0.5f*disk_sample.y*camera_lens_diameter);
    }

    //*** ray direction is line from lens sample to p ***//

    float3 ray_direction = p - ray_origin;

    //*** rotate ray origin and direction into the direction of the camera view *** //

    ray_origin = d_rotatePoint( ray_origin, -0.5*M_PI+camera_direction.x, 0.5f*M_PI-camera_direction.y ) + camera_position;

    ray_direction = d_rotatePoint( ray_direction, -0.5*M_PI+camera_direction.x, 0.5f*M_PI-camera_direction.y );
    ray_direction /= d_magnitude(ray_direction);

    optix::Ray ray;

    prd.strength = 1.f/float(launch_dim.x);

    prd.area = 1.f/float(launch_dim.x);
    prd.origin_UUID = origin_ID;
    prd.periodic_depth = 0;
    prd.face = 1;
    prd.source_ID = 0;

    ray = optix::make_Ray(ray_origin, ray_direction, camera_ray_type, 1e-5, RT_DEFAULT_MAX);

    rtTrace( top_object, ray, prd);

}

RT_PROGRAM void pixel_label_raygen(){

    uint indx = launch_dim.y*launch_index.z + launch_index.y;

    optix::int2 camera_resolution = optix::make_int2( launch_dim.y, launch_dim.z );

    PerRayData prd;
    prd.seed = tea<16>(indx,random_seed);

    float3 sp;

    uint ii = launch_index.y; //x-pixel

    uint jj = launch_index.z; //y-pixel

    size_t origin_ID = jj*launch_dim.y + ii; //global pixel index

    // Map sample to center of pixel
    sp.y = (-0.5f + (ii+0.5f)/float(camera_resolution.x))*sensor_size.x;
    sp.z = (0.5f - (jj+0.5f)/float(camera_resolution.y))*sensor_size.y;
    sp.x = camera_viewplane_length;



    // *** Determine point 'p' on focal plane that passes through the lens center (0,0) and pixel sample (view direction coordinate aligned) *** //

    float3 p = make_float3(camera_focal_length,sp.y/camera_viewplane_length*camera_focal_length,sp.z/camera_viewplane_length*camera_focal_length);

    // *** Ray is launched from center of lens *** //

    float3 ray_origin = make_float3(0,0,0);

    //*** ray direction is line from ray origin to p ***//

    float3 ray_direction = p;

    //*** rotate ray origin and direction into the direction of the camera view *** //

    ray_origin = d_rotatePoint( ray_origin, -0.5*M_PI+camera_direction.x, 0.5f*M_PI-camera_direction.y ) + camera_position;

    ray_direction = d_rotatePoint( ray_direction, -0.5*M_PI+camera_direction.x, 0.5f*M_PI-camera_direction.y );
    ray_direction /= d_magnitude(ray_direction);

    optix::Ray ray;

    prd.strength = 1.f;

    prd.area = 1.f;
    prd.origin_UUID = origin_ID;
    prd.periodic_depth = 0;
    prd.face = 1;
    prd.source_ID = 0;

    ray = optix::make_Ray(ray_origin, ray_direction, pixel_label_ray_type, 1e-5, RT_DEFAULT_MAX);

    rtTrace( top_object, ray, prd);

}
