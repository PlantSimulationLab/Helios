/** \file "rayHit.cu" Functions for object intersection.
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
#include <optixu/optixu_math_namespace.h>

#include "RayTracing.cu.h"

using namespace optix; 

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(PerRayData, prd, rtPayload, );
rtDeclareVariable(uint3, launch_dim,   rtLaunchDim, );
rtDeclareVariable(rtObject,      top_object, , );

rtDeclareVariable( unsigned int, UUID, attribute UUID, );

rtDeclareVariable(float, diffuseFlux,, );

rtDeclareVariable(float, diffuse_extinction,, );
rtDeclareVariable(float3, diffuse_peak_dir,, );
rtDeclareVariable(float, diffuse_dist_norm,, );

rtDeclareVariable(float2, periodic_flag,, );

rtBuffer<float, 1> rho, tau, eps;

rtBuffer<unsigned int, 1> primitive_type;

rtBuffer<float, 1>   radiation_in;
rtBuffer<float, 1>   radiation_out_top;
rtBuffer<float, 1>   radiation_out_bottom;
rtBuffer<float, 1>   scatter_buff_top;
rtBuffer<float, 1>   scatter_buff_bottom;
rtBuffer<bool, 1> twosided_flag;

rtBuffer<float, 1>   Rsky;

rtBuffer<float, 2>  transform_matrix;

rtBuffer<float3, 2> bbox_vertices;

rtBuffer<uint, 1> objectID;

RT_PROGRAM void closest_hit_direct(){

  uint objID = objectID[UUID];

  if( (periodic_flag.x==1 || periodic_flag.y==1) && primitive_type[objID] == 5 ){ //periodic boundary condition

    float3 ray_origin = ray.origin + t_hit*ray.direction;

    float eps=0;

    if( prd.periodic_depth<9 ){
      if( periodic_flag.x==1 ){

	float2 xbounds = make_float2(bbox_vertices[make_uint2(0,0)].x,bbox_vertices[make_uint2(1,1)].x);
	if(ray_origin.x==xbounds.x ){//-x facing boundary
	  ray_origin.x = xbounds.y - eps;
	}else if( ray_origin.x==xbounds.y ){//+x facing boundary
	  ray_origin.x = xbounds.x + eps;
	}
      
      }
      if( periodic_flag.y==1 ){
	
	float2 ybounds = make_float2(bbox_vertices[make_uint2(0,0)].y,bbox_vertices[make_uint2(1,1)].y);
	if( ray_origin.y==ybounds.x ){//-y facing boundary
	  ray_origin.y = ybounds.y - eps;
	}else if( ray_origin.y==ybounds.y ){//+y facing boundary
	  ray_origin.y = ybounds.x + eps;
	}
      
      }
    }
    
    float3 ray_direction = ray.direction;
    
    optix::Ray ray_periodic = optix::make_Ray(ray_origin, ray_direction, ray.ray_type, 1e-4, RT_DEFAULT_MAX);
    PerRayData prd_periodic = prd;
    prd_periodic.periodic_depth++;
  
    rtTrace( top_object, ray_periodic, prd_periodic );

  }
  
};

RT_PROGRAM void closest_hit_diffuse()
{

  uint origin_UUID = prd.origin_UUID;

  uint objID = objectID[UUID];

  if( (periodic_flag.x==1 || periodic_flag.y==1) && primitive_type[objID] == 5 ){ //periodic boundary condition

    float3 ray_origin = ray.origin + t_hit*ray.direction;

    float eps=0;

    if( prd.periodic_depth<9 ){
      if( periodic_flag.x==1 ){

	float2 xbounds = make_float2(bbox_vertices[make_uint2(0,0)].x,bbox_vertices[make_uint2(1,1)].x);
	
	if( ray_origin.x==xbounds.x ){//-x facing boundary
	  ray_origin.x = xbounds.y - eps;
	}else if( ray_origin.x==xbounds.y ){//+x facing boundary
	  ray_origin.x = xbounds.x + eps;
	}
	
      }
      if( periodic_flag.y==1 ){
	
	float2 ybounds = make_float2(bbox_vertices[make_uint2(0,0)].y,bbox_vertices[make_uint2(1,1)].y);
	if( ray_origin.y==ybounds.x ){//-y facing boundary
	  ray_origin.y = ybounds.y - eps;
	}else if( ray_origin.y==ybounds.y ){//+y facing boundary
	  ray_origin.y = ybounds.x + eps;
	}
	
      }
    }
    
    float3 ray_direction = ray.direction;
    
    optix::Ray ray_periodic = optix::make_Ray(ray_origin, ray_direction, ray.ray_type, 1e-4, RT_DEFAULT_MAX);
    PerRayData prd_periodic = prd;
    prd_periodic.periodic_depth++;

    rtTrace( top_object, ray_periodic, prd_periodic );

  }else{

  //Note: UUID corresponds to the object that the ray hit (i.e., where energy is coming from), and UUID_origin is the object the ray originated from (i.e., where the energy is being recieved)

  float t_rho = rho[origin_UUID];
  float t_tau = tau[origin_UUID];

  //find out if we hit top or bottom surface
  float3 normal;

  float m[16];
  for( uint i=0; i<16; i++ ){
    m[i] = transform_matrix[ optix::make_uint2(i,objID) ];
  }

  if( primitive_type[objID] == 0 || primitive_type[objID] == 3 ){//hit patch or tile
    float3 s0 = make_float3(0,0,0);
    float3 s1 = make_float3(1,0,0);
    float3 s2 = make_float3(0,1,0);
    s0 = d_transformPoint(m,s0);
    s1 = d_transformPoint(m,s1);
    s2 = d_transformPoint(m,s2);
    normal = cross(s1-s0,s2-s0);
  }else if( primitive_type[UUID] == 1 ){//hit triangle
    float3 v0 = d_transformPoint(m,make_float3(0,0,0));
    float3 v1 = d_transformPoint(m,make_float3(0,1,0));
    float3 v2 = d_transformPoint(m,make_float3(1,1,0));
    normal = cross(v1-v0,v2-v1);
  }else if( primitive_type[UUID] == 2 ){//hit disk
    float3 v0 = d_transformPoint(m,make_float3(0,0,0));
    float3 v1 = d_transformPoint(m,make_float3(1,0,0));
    float3 v2 = d_transformPoint(m,make_float3(0,1,0));
    normal = cross(v1-v0,v2-v0);
  }else if( primitive_type[UUID] == 4 ){//hit voxel
    float3 vmin = d_transformPoint(m,make_float3(-0.5,-0.5,-0.5));
    float3 vmax = d_transformPoint(m,make_float3(0.5,0.5,0.5));
  }
  normal = normalize(normal);
  
  bool face = dot(normal,ray.direction)<0;

  double strength;
  if( face || primitive_type[objID] == 4 ){
    strength = radiation_out_top[UUID]*prd.strength;
  }else{
    strength = radiation_out_bottom[UUID]*prd.strength;
  }

  if( primitive_type[objectID[origin_UUID]] == 4 ){ //ray was launched from voxel

    float kappa = t_rho; //just a reminder that rho is actually the absorption coefficient
    float sigma_s = t_tau; //just a reminder that tau is actually the scattering coefficient
    float beta = kappa+sigma_s;

    // absorption
    atomicAdd( &radiation_in[origin_UUID], strength*exp(-beta*0.5*prd.area)*kappa/beta );
    
    // scattering
    atomicAdd( &scatter_buff_top[origin_UUID], strength*exp(-beta*0.5*prd.area)*sigma_s/beta );
    
  }else{ //ray was NOT launched from voxel
    
    // absorption
    atomicAdd( &radiation_in[origin_UUID], strength*(1.f-t_rho-t_tau) );
    
    if( (t_rho>0 || t_tau>0) && strength>0 ){
      
      if( prd.face ){//reflection from top, transmission from bottom
  	atomicFloatAdd( &scatter_buff_top[origin_UUID], strength*t_rho ); //reflection
  	atomicFloatAdd( &scatter_buff_bottom[origin_UUID], strength*t_tau ); //transmission
      }else{//reflection from bottom, transmission from top
  	atomicFloatAdd( &scatter_buff_bottom[origin_UUID], strength*t_rho ); //reflection
  	atomicFloatAdd( &scatter_buff_top[origin_UUID], strength*t_tau ); //transmission
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

RT_PROGRAM void miss_direct(){

  uint objID = objectID[prd.origin_UUID];

  float t_rho = rho[prd.origin_UUID];
  float t_tau = tau[prd.origin_UUID];

  //absorption
  atomicAdd( &radiation_in[prd.origin_UUID], prd.strength*(1.f-t_rho-t_tau) );

  if( t_rho>0 || t_tau>0 ){

    if( prd.face ){//reflection from top, transmission from bottom
      atomicFloatAdd( &scatter_buff_top[prd.origin_UUID], prd.strength*t_rho ); //reflection
      atomicFloatAdd( &scatter_buff_bottom[prd.origin_UUID], prd.strength*t_tau ); //transmission
    }else{//reflection from bottom, transmission from top
      atomicFloatAdd( &scatter_buff_bottom[prd.origin_UUID], prd.strength*t_rho ); //reflection
      atomicFloatAdd( &scatter_buff_top[prd.origin_UUID], prd.strength*t_tau ); //transmission
    }
  }
  
}

RT_PROGRAM void miss_diffuse(){
  
  double strength;
  if( prd.face || primitive_type[objectID[prd.origin_UUID] ] == 3 ){
    strength = radiation_out_top[prd.origin_UUID]*prd.strength*prd.area;
  }else{
    strength = radiation_out_bottom[prd.origin_UUID]*prd.strength*prd.area;
  }
  
  float t_rho = rho[prd.origin_UUID];
  float t_tau = tau[prd.origin_UUID];

  atomicFloatAdd( &Rsky[prd.origin_UUID], strength);

  if( primitive_type[ objectID[prd.origin_UUID] ] == 4 ){ //ray was launched from voxel

    float kappa = t_rho; //just a reminder that rho is actually the absorption coefficient
    float sigma_s = t_tau; //just a reminder that tau is actually the scattering coefficient
    float beta = kappa+sigma_s;

    // absorption
    atomicAdd( &radiation_in[prd.origin_UUID], diffuseFlux*prd.strength*kappa/beta );

    // scattering
    atomicAdd( &scatter_buff_top[prd.origin_UUID], diffuseFlux*prd.strength*sigma_s/beta );

  }else{ //ray was NOT launched from voxel

    float fd = 1.f;
    if( diffuse_extinction>0.f ){
      float psi = acos_safe( dot(diffuse_peak_dir,ray.direction) );
      if( psi<M_PI/180.f ){
	fd = pow(M_PI/180.f,-diffuse_extinction)*diffuse_dist_norm;
      }else{
	fd = pow(psi,-diffuse_extinction)*diffuse_dist_norm;
      }
    }
  
    //absorption
    atomicAdd( &radiation_in[prd.origin_UUID], fd*diffuseFlux*(1.f-t_rho-t_tau)*prd.strength );

    if( t_rho>0 || t_tau>0 ){

      if( prd.face ){//reflection from top, transmission from bottom
	atomicFloatAdd( &scatter_buff_top[prd.origin_UUID], fd*diffuseFlux*t_rho*prd.strength ); //reflection
	atomicFloatAdd( &scatter_buff_bottom[prd.origin_UUID], fd*diffuseFlux*t_tau*prd.strength ); //transmission
      }else{//reflection from bottom, transmission from top
	atomicFloatAdd( &scatter_buff_bottom[prd.origin_UUID], fd*diffuseFlux*t_rho*prd.strength ); //reflection
	atomicFloatAdd( &scatter_buff_top[prd.origin_UUID], fd*diffuseFlux*t_tau*prd.strength ); //transmission
      }
    }

  }

}
