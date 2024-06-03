/** \file primitiveIntersection.cu "File containing OptiX ray-primitive intersection and bounding-box programs"

    Copyright (C) 2016-2023 Brian Bailey

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

//--- Tiles ---//
rtBuffer<float3, 2> tile_vertices;
rtBuffer<unsigned int, 1> tile_UUID;

//--- Voxels ---//
rtBuffer<float3, 2> voxel_vertices;
rtBuffer<unsigned int, 1> voxel_UUID;

//--- Bounding Box ---//
rtBuffer<float3, 2> bbox_vertices;
rtBuffer<unsigned int, 1> bbox_UUID;

//--- Masks ---//
rtBuffer<bool, 3>   maskdata;
rtBuffer<int2, 1>   masksize;
rtBuffer<int, 1>   maskID;

rtBuffer<float2, 2> uvdata;
rtBuffer<int, 1> uvID;

rtBuffer<uint,1> objectID;

rtBuffer<int2,1> object_subdivisions;

rtBuffer<char, 1> twosided_flag;

//----------------- Rectangle Primitive ----------------------//

/** OptiX ray-rectangle intersection program. */

RT_PROGRAM void rectangle_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

  if( prd.origin_UUID == patch_UUID[objID] ){ //the ray should not intersect the primitive from which it was launched
    return;
  }else if( twosided_flag[ patch_UUID[objID] ]>=2 ){ //if twosided_flag=2, ignore intersection (transparent)
      return;
  }
  
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

	uint ID = objectID[U];
	
	if( maskID[ID]==-1 ){ //no texture transparency
 	  if( rtPotentialIntersection( t ) ) {
	    UUID = patch_UUID[objID];
	    rtReportIntersection(0);
	  }	  
	}else{ 	//use transparency mask

	  float amag = d_magnitude(a);
	  float bmag = d_magnitude(b);
	  float2 uv = make_float2( ddota/amag/amag, 1.f-ddotb/bmag/bmag );
	  int2 sz = masksize[ maskID[ID] ];
	  uint3 ind;
	  if( uvID[ID]==-1 ){ //does not have custom (u,v) coordinates
	    ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), maskID[ID] );
	  }else{
	    
	    float2 uvmin = uvdata[ make_uint2(1,uvID[ID]) ];
	    float2 duv;
	    duv.x = uvdata[ make_uint2(0,uvID[ID]) ].x - uvdata[ make_uint2(1,uvID[ID]) ].x;
	    duv.y = uvdata[ make_uint2(2,uvID[ID]) ].y - uvdata[ make_uint2(1,uvID[ID]) ].y;
	    
	    ind = make_uint3( roundf(float(sz.x-1)*(uvmin.x+fabs(uv.x)*duv.x)), roundf(float(sz.y-1)*(uvmin.y+fabs(uv.y)*duv.y)), maskID[ID] );
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

    if( prd.origin_UUID == triangle_UUID[objID] ){ //the ray should not intersect the primitive from which it was launched
        return;
    }else if( twosided_flag[ triangle_UUID[objID] ]>=2 ){ //if twosided_flag=2, ignore intersection (transparent)
        return;
    }

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

	uint ID = objectID[U];

    if( maskID[ID]==-1 ){ //no texture transparency
	  if( rtPotentialIntersection( t ) ) {
	    UUID = triangle_UUID[objID];
	    rtReportIntersection(0);
	  }
	}else{ //has texture transparency

	  int2 sz = masksize[ maskID[ID] ];

	  float2 uv0 = uvdata[ make_uint2(0,uvID[ID]) ];
	  float2 uv1 = uvdata[ make_uint2(1,uvID[ID]) ];
	  float2 uv2 = uvdata[ make_uint2(2,uvID[ID]) ];

	  float2 uv = uv0 + beta*(uv1-uv0) + gamma*(uv2-uv0);
	  uv.y = 1.f-uv.y;

	  uint3 ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), maskID[ID] );
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

    if( prd.origin_UUID == disk_UUID[objID] ){ //the ray should not intersect the primitive from which it was launched
        return;
    }else if( twosided_flag[ disk_UUID[objID] ]>=2 ){ //if twosided_flag=2, ignore intersection (transparent)
        return;
    }

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

//----------------- Voxel Primitive ----------------------//

/** OptiX ray-voxel intersection program. */

RT_PROGRAM void voxel_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

    if( prd.origin_UUID == voxel_UUID[objID] ){ //the ray should not intersect the primitive from which it was launched
        return;
    }else if( twosided_flag[ voxel_UUID[objID] ]>=2 ){ //if twosided_flag=2, ignore intersection (transparent)
        return;
    }

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

//----------------- Bounding Box Primitive ----------------------//

/** OptiX ray-rectangle intersection program. */

RT_PROGRAM void bbox_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

    if( prd.origin_UUID == bbox_UUID[objID] ){ //the ray should not intersect the primitive from which it was launched
        return;
    }else if( twosided_flag[ bbox_UUID[objID] ]>=2 ){ //if twosided_flag=2, ignore intersection (transparent)
        return;
    }

  float3 v0 = bbox_vertices[ make_uint2(0, objID) ];
  float3 v1 = bbox_vertices[ make_uint2(1, objID) ];
  float3 v2 = bbox_vertices[ make_uint2(2, objID) ];
  float3 v3 = bbox_vertices[ make_uint2(3, objID) ];

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
	
	if( rtPotentialIntersection( t ) ) {
	  UUID = bbox_UUID[objID];
	  rtReportIntersection(0);
	}	  

      }
    }
  }

}

/** Axis-aligned bounding box program for bounding box rectangle primitives. */

RT_PROGRAM void bbox_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 v0 = bbox_vertices[ make_uint2(0, objID) ];
  float3 v1 = bbox_vertices[ make_uint2(1, objID) ];
  float3 v2 = bbox_vertices[ make_uint2(2, objID) ];
  float3 v3 = bbox_vertices[ make_uint2(3, objID) ];
  float3 min = make_float3( fmin(fmin(v0.x,v1.x),fmin(v2.x,v3.x)), fmin(fmin(v0.y,v1.y),fmin(v2.y,v3.y)), fmin(fmin(v0.z,v1.z),fmin(v2.z,v3.z)) );
  float3 max = make_float3( fmax(fmax(v0.x,v1.x),fmax(v2.x,v3.x)), fmax(fmax(v0.y,v1.y),fmax(v2.y,v3.y)), fmax(fmax(v0.z,v1.z),fmax(v2.z,v3.z)) );
  aabb->set(min, max);
}

//----------------- Tile Object ----------------------//

/** OptiX ray-tile intersection program. */

RT_PROGRAM void tile_intersect(int objID /**< [in] index of primitive in geometric object.*/)
{

    if( prd.origin_UUID == tile_UUID[objID] ){ //the ray should not intersect the primitive from which it was launched
        return;
    }else if( twosided_flag[ tile_UUID[objID] ]>=2 ){ //if twosided_flag=2, ignore intersection (transparent)
        return;
    }

  float3 v0 = tile_vertices[ make_uint2(0, objID) ];
  float3 v1 = tile_vertices[ make_uint2(1, objID) ];
  float3 v2 = tile_vertices[ make_uint2(2, objID) ];
  float3 v3 = tile_vertices[ make_uint2(3, objID) ];

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

	float amag = d_magnitude(a);
	float bmag = d_magnitude(b);
	float2 uv = make_float2( 1.f-ddota/amag/amag, ddotb/bmag/bmag );

	uint U = tile_UUID[objID];

	uint ID = objectID[U];
	
	if( maskID[ID]==-1 ){ //no texture transparency
	  if( rtPotentialIntersection( t ) ) {
	    //UUID = tile_UUID[objID]+floor(uv.y*object_subdivisions[objID].y)*object_subdivisions[objID].x+floor((1.f-uv.x)*object_subdivisions[objID].x);
	    UUID = U+floor(uv.y*object_subdivisions[ID].y)*object_subdivisions[ID].x+floor((1.f-uv.x)*object_subdivisions[ID].x);
	    rtReportIntersection(0);
	  }	  
	}else{ 	//use transparency mask

	 
	  int2 sz = masksize[ maskID[ID] ];
	  uint3 ind = make_uint3( roundf(float(sz.x-1)*fabs(uv.x)), roundf(float(sz.y-1)*fabs(uv.y)), maskID[ID] );

	  if( maskdata[ind] ){
	    if( rtPotentialIntersection( t ) ) {
	      UUID = tile_UUID[objID]+floor(uv.y*object_subdivisions[ID].y)*object_subdivisions[ID].x+floor((1.f-uv.x)*object_subdivisions[ID].x);
	      rtReportIntersection(0);
	    }
	    
	  }
	
	}
      }
    }
  }

}

/** Axis-aligned bounding box program for tile objects. */

RT_PROGRAM void tile_bounds (int objID, float result[6])
{
  optix::Aabb* aabb = (optix::Aabb*)result;
  float3 v0 = tile_vertices[ make_uint2(0, objID) ];
  float3 v1 = tile_vertices[ make_uint2(1, objID) ];
  float3 v2 = tile_vertices[ make_uint2(2, objID) ];
  float3 v3 = tile_vertices[ make_uint2(3, objID) ];
  float3 min = make_float3( fmin(fmin(v0.x,v1.x),fmin(v2.x,v3.x)), fmin(fmin(v0.y,v1.y),fmin(v2.y,v3.y)), fmin(fmin(v0.z,v1.z),fmin(v2.z,v3.z)) );
  float3 max = make_float3( fmax(fmax(v0.x,v1.x),fmax(v2.x,v3.x)), fmax(fmax(v0.y,v1.y),fmax(v2.y,v3.y)), fmax(fmax(v0.z,v1.z),fmax(v2.z,v3.z)) );
  aabb->set(min, max);
}
