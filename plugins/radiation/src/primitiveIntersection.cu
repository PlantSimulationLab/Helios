/** \file primitiveIntersection.cu "File containing OptiX ray-primitive intersection and bounding-box programs" 
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

#include <optix.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>

using namespace optix;

#include "RayTracing.cu.h"

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData, prd, rtPayload, );

rtDeclareVariable( unsigned int, UUID, attribute UUID, );

//--- Patches ---//
rtBuffer<float3, 2> patch_vertices;
rtBuffer<unsigned int, 1> patch_UUID;

//--- Triangles ---//
rtBuffer<float3, 2> triangle_vertices;
rtBuffer<unsigned int, 1> triangle_UUID;

//--- Disks ---//
//To-Do: disks not finished
rtBuffer<unsigned int, 1> disk_UUID;

//--- AlphaMasks ---//
rtBuffer<float3, 2> alphamask_vertices;
rtBuffer<unsigned int, 1> alphamask_UUID;

//--- Voxels ---//
rtBuffer<float3, 2> voxel_vertices;
rtBuffer<unsigned int, 1> voxel_UUID;

//--- Masks ---//
rtBuffer<bool, 3>   maskdata;
rtBuffer<int2, 1>   masksize;
rtBuffer<int, 1>   maskID;

rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;

//----------------- Rectangle Primitive ----------------------//

/** OptiX ray-rectangle intersection program. */

RT_PROGRAM void rectangle_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

  float3 v0 = patch_vertices[ make_uint2(0, objID) ];
  float3 v1 = patch_vertices[ make_uint2(1, objID) ];
  float3 v2 = patch_vertices[ make_uint2(2, objID) ];
  float3 v3 = patch_vertices[ make_uint2(3, objID) ];

  float3 anchor = v0;
  float3 normal = normalize( cross( v1-v0, v2-v0 ) );

  float3 a = v1-v0;
  float3 b = v3-v0;
  
  float t = dot(anchor - ray.origin, normal) / dot(ray.direction, normal); 

  if( t==t && t>1e-8 && t<1e8 ){
			
    float3 p = ray.origin + ray.direction * t;
    float3 d = p - anchor;
	
    float ddota = dot(d,a);
	
    if (ddota > 0.0 && ddota < dot(a,a) ){
		
      float ddotb = dot(d, b);
  
      if (ddotb > 0.0 && ddotb < dot(b,b) ){

	uint U = patch_UUID[objID];
	
	if( maskID[U]==-1 ){ //no texture transparency
	  if( rtPotentialIntersection( t ) ) {
	    UUID = patch_UUID[objID];
	    rtReportIntersection(0);
	  }	  
	}else{ 	//use transparency mask

	  float amag = d_magnitude(a);
	  float bmag = d_magnitude(b);
	  float2 uv = make_float2( ddota/amag/amag, 1.f-ddotb/bmag/bmag );
	  uint ID = maskID[U];
	  int2 sz = masksize[ID];
	  uint3 ind;
	  if( uvID[U]==-1 ){ //does not have custom (u,v) coordinates
	    ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), ID );
	  }else{
	    float2 uvmin = uvdata[ make_uint2(3,uvID[U]) ];
	    float2 duv;
	    duv.x = uvdata[ make_uint2(1,uvID[U]) ].x - uvdata[ make_uint2(0,uvID[U]) ].x;
	    duv.y = uvdata[ make_uint2(1,uvID[U]) ].y - uvdata[ make_uint2(2,uvID[U]) ].y;
	    ind = make_uint3( roundf(float(sz.x-1)*(uvmin.x+fabs(uv.x)*duv.x)), roundf(float(sz.y-1)*(uvmin.y+fabs(uv.y)*duv.y)), ID );
	    if( ind.x>=sz.x || ind.y>=sz.y ){
	      rtPrintf("ERROR: texture out of bounds.\n");
	    }
	  }
	  if( maskdata[ind] ){
	    if( rtPotentialIntersection( t ) ) {
	      UUID = patch_UUID[objID];
	      rtReportIntersection(0);
	    }
	    
	  }
	
	}
      }
    }
  }

}

/** Axis-aligned bounding box program for rectangle primitives. */

RT_PROGRAM void rectangle_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 v0 = patch_vertices[ make_uint2(0, objID) ];
  float3 v1 = patch_vertices[ make_uint2(1, objID) ];
  float3 v2 = patch_vertices[ make_uint2(2, objID) ];
  float3 v3 = patch_vertices[ make_uint2(3, objID) ];
  float3 min = make_float3( fmin(fmin(v0.x,v1.x),fmin(v2.x,v3.x)), fmin(fmin(v0.y,v1.y),fmin(v2.y,v3.y)), fmin(fmin(v0.z,v1.z),fmin(v2.z,v3.z)) );
  float3 max = make_float3( fmax(fmax(v0.x,v1.x),fmax(v2.x,v3.x)), fmax(fmax(v0.y,v1.y),fmax(v2.y,v3.y)), fmax(fmax(v0.z,v1.z),fmax(v2.z,v3.z)) );
  aabb->set(min, max);
}

//----------------- Triangle Primitive ----------------------//

/** OptiX ray-triangle intersection program. */

RT_PROGRAM void triangle_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

  float3 v0 = triangle_vertices[ make_uint2(0, objID) ];
  float3 v1 = triangle_vertices[ make_uint2(1, objID) ];
  float3 v2 = triangle_vertices[ make_uint2(2, objID) ];
  
  float a = v0.x - v1.x, b = v0.x - v2.x, c = ray.direction.x, d = v0.x - ray.origin.x; 
  float e = v0.y - v1.y, f = v0.y - v2.y, g = ray.direction.y, h = v0.y - ray.origin.y;
  float i = v0.z - v1.z, j = v0.z - v2.z, k = ray.direction.z, l = v0.z - ray.origin.z;
		
  float m = f * k - g * j, n = h * k - g * l, p = f * l - h * j;
  float q = g * i - e * k, s = e * j - f * i;
  
  float inv_denom  = 1.f / (a * m + b * q + c * s);
  
  float e1 = d * m - b * n - c * p;
  float beta = e1 * inv_denom;
  
  if (beta > 0.0){

    float r = r = e * l - h * i;
    float e2 = a * n + d * q + c * r;
    float gamma = e2 * inv_denom;

    if (gamma > 0.0 && beta + gamma < 1.0 ){

      float e3 = a * p - b * r + d * s;
      float t = e3 * inv_denom;

      if( t>1e-8 ){

	uint U = triangle_UUID[objID];

	if( maskID[U]==-1 ){ //no texture transparency
	  if( rtPotentialIntersection( t ) ) {
	    UUID = triangle_UUID[objID];
	    rtReportIntersection(0);
	  }
	}else{ //has texture transparency

	  uint ID = maskID[U];
	  int2 sz = masksize[ID];

	  float2 uv0 = uvdata[ make_uint2(0,uvID[U]) ];
	  float2 uv1 = uvdata[ make_uint2(1,uvID[U]) ];
	  float2 uv2 = uvdata[ make_uint2(2,uvID[U]) ];

	  float2 uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0);

	  uint3 ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), ID );
	  if( ind.x>=sz.x || ind.y>=sz.y ){
	    rtPrintf("ERROR: texture out of bounds.\n");
	  }
	  if( maskdata[ind] ){
	    if( rtPotentialIntersection( t ) ) {
	      UUID = triangle_UUID[objID];
	      rtReportIntersection(0);
	    }
	    
	  }

	}
	
      }

    }

  }


}


/** Axis-aligned bounding box program for triangle primitives. */
RT_PROGRAM void triangle_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 v0 = triangle_vertices[ make_uint2(0, objID) ];
  float3 v1 = triangle_vertices[ make_uint2(1, objID) ];
  float3 v2 = triangle_vertices[ make_uint2(2, objID) ];
  float3 mn = make_float3( fmin(fmin(v0.x,v1.x),v2.x), fmin(fmin(v0.y,v1.y),v2.y), fmin(fmin(v0.z,v1.z),v2.z) );
  float3 mx = make_float3( fmax(fmax(v0.x,v1.x),v2.x), fmax(fmax(v0.y,v1.y),v2.y), fmax(fmax(v0.z,v1.z),v2.z) );
  aabb->set(mn, mx);
}

//----------------- Disk Primitive ----------------------//

/** OptiX ray-disk intersection program. */

RT_PROGRAM void disk_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

  float3 center = make_float3(0,0,0);
  float3 normal = make_float3(0,0,1);

  float t = dot(center-ray.origin,normal)/dot(ray.direction,normal);

  if( t>1e-6 ){

    float3 p = ray.origin + t*ray.direction;

    float3 r = p - center;

    if( r.x*r.x+r.y*r.y+r.z*r.z < 1.f ){

  	if( rtPotentialIntersection( t ) ) {
	  UUID = disk_UUID[objID];
  	  rtReportIntersection(0);
  	} 

    }	

  }

}


/** Axis-aligned bounding box program for disk primitives. */
RT_PROGRAM void disk_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 mn = make_float3( -1, -1, -0.001f );
  float3 mx = make_float3( 1, 1, 0.001f );
  aabb->set(mn, mx);
}

//----------------- AlphaMask Primitive ----------------------//

/** OptiX ray-Alphamask intersection program. */

RT_PROGRAM void alphamask_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

  float3 v0 = alphamask_vertices[ make_uint2(0, objID) ];
  float3 v1 = alphamask_vertices[ make_uint2(1, objID) ];
  float3 v2 = alphamask_vertices[ make_uint2(2, objID) ];
  float3 v3 = alphamask_vertices[ make_uint2(3, objID) ];

  float3 anchor = v0;
  float3 normal = normalize( cross( v1-v0, v2-v0 ) );

  float3 a = v1-v0;
  float3 b = v3-v0;

  float t = dot(anchor - ray.origin, normal) / dot(ray.direction, normal);

  if( t==t && t>1e-8 && t<1e8 ){
			
    float3 p = ray.origin + ray.direction * t;
    float3 d = p - anchor;
	
    float ddota = dot(d,a);
	
    if (ddota > 0.0 && ddota < dot(a,a) ){
		
      float ddotb = dot(d, b);
  
      if (ddotb > 0.0 && ddotb < dot(b,b) ){	

	//use mask
	float amag = d_magnitude(a);
	float bmag = d_magnitude(b);
	float2 uv = make_float2( ddota/amag/amag, 1.f-ddotb/bmag/bmag );
	uint ID = maskID[ alphamask_UUID[objID] ];
	int2 sz = masksize[ID];
	uint3 ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), ID );
	if( maskdata[ind] ){
	  if( rtPotentialIntersection( t ) ) {
	    UUID = alphamask_UUID[objID];
	    rtReportIntersection(0);
	  }

	}

      }
    }
  }

}

/** Axis-aligned bounding box program for AlphaMask primitives. */

RT_PROGRAM void alphamask_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 v0 = alphamask_vertices[ make_uint2(0, objID) ];
  float3 v1 = alphamask_vertices[ make_uint2(1, objID) ];
  float3 v2 = alphamask_vertices[ make_uint2(2, objID) ];
  float3 v3 = alphamask_vertices[ make_uint2(3, objID) ];
  float3 min = make_float3( fmin(fmin(v0.x,v1.x),fmin(v2.x,v3.x)), fmin(fmin(v0.y,v1.y),fmin(v2.y,v3.y)), fmin(fmin(v0.z,v1.z),fmin(v2.z,v3.z)) );
  float3 max = make_float3( fmax(fmax(v0.x,v1.x),fmax(v2.x,v3.x)), fmax(fmax(v0.y,v1.y),fmax(v2.y,v3.y)), fmax(fmax(v0.z,v1.z),fmax(v2.z,v3.z)) );
  aabb->set(min, max);
}

//----------------- Voxel Primitive ----------------------//

/** OptiX ray-voxel intersection program. */

RT_PROGRAM void voxel_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

  float x0 = voxel_vertices[ make_uint2(0, objID) ].x;
  float y0 = voxel_vertices[ make_uint2(0, objID) ].y;
  float z0 = voxel_vertices[ make_uint2(0, objID) ].z;

  float x1 = voxel_vertices[ make_uint2(1, objID) ].x;
  float y1 = voxel_vertices[ make_uint2(1, objID) ].y;
  float z1 = voxel_vertices[ make_uint2(1, objID) ].z;

  float ox = ray.origin.x; float oy = ray.origin.y; float oz = ray.origin.z;
  float dx = ray.direction.x; float dy = ray.direction.y; float dz = ray.direction.z;
	
  float tx_min, ty_min, tz_min;
  float tx_max, ty_max, tz_max; 

  float a = 1.0 / dx;
  if (a >= 0) {
    tx_min = (x0 - ox) * a;
    tx_max = (x1 - ox) * a;
  }
  else {
    tx_min = (x1 - ox) * a;
    tx_max = (x0 - ox) * a;
  }
	
  float b = 1.0 / dy;
  if (b >= 0) {
    ty_min = (y0 - oy) * b;
    ty_max = (y1 - oy) * b;
  }
  else {
    ty_min = (y1 - oy) * b;
    ty_max = (y0 - oy) * b;
  }
  
  float c = 1.0 / dz;
  if (c >= 0) {
    tz_min = (z0 - oz) * c;
    tz_max = (z1 - oz) * c;
  }
  else {
    tz_min = (z1 - oz) * c;
    tz_max = (z0 - oz) * c;
  }
  
  float t0, t1;
	
  // find largest entering t value
  
  if (tx_min > ty_min)
    t0 = tx_min;
  else
    t0 = ty_min;
  
  if (tz_min > t0)
    t0 = tz_min;	
  
  // find smallest exiting t value
  
  if (tx_max < ty_max)
    t1 = tx_max;
  else
    t1 = ty_max;
  
  if (tz_max < t1)
    t1 = tz_max;
  
  if (t0 < t1 && t0 > 1e-5){//note: if ray originated inside voxel, no intersection occurs.
    if( rtPotentialIntersection( t0 ) ) {
      prd.area = fabs(t1-t0);
      UUID = voxel_UUID[objID];
      rtReportIntersection(0);
    }
  }

}

/** Axis-aligned bounding box program for voxel primitives. */

RT_PROGRAM void voxel_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 min = voxel_vertices[ make_uint2(0, objID) ];
  float3 max = voxel_vertices[ make_uint2(1, objID) ];
  aabb->set(min, max);
}
