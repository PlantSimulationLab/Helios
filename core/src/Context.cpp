/** \file "Context.cpp" Context declarations. 
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

#include "Context.h"

using namespace helios;

Context::Context(void){

  //---- ALL DEFAULT VALUES ARE SET HERE ----//

  iscontextinitialized=1;

  sim_date = make_Date(1,6,2000);

  sim_time = make_Time(12,0);

   // --- Initialize random number generator ---- //

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);

  // --- Set Geometry as `Clean' --- //

  isgeometrydirty = false;

  currentUUID = 0;

}

int Context::selfTest(void){

  std::cout << "Running Context self-test..." << std::flush;

  Context context_test;

  uint error_count=0;

  double errtol = 1e-7;

  std::string answer;

  //------- Add Patch --------//

  vec3 center,center_r;
  vec2 size, size_r;
  std::vector<vec3> vertices, vertices_r;
  SphericalCoord rotation, rotation_r;
  vec3 normal, normal_r;
  RGBcolor color, color_r;
  uint UUID;
  std::vector<uint> UUIDs;
  PrimitiveType type;
  float area_r;
  Primitive* prim;

  //uint addPatch( const vec3& center, const vec2& size );
  center = make_vec3(1,2,3);
  size = make_vec2(1,2);
  vertices.resize(4);
  vertices.at(0) = center + make_vec3( -0.5f*size.x, -0.5f*size.y, 0.f);
  vertices.at(1) = center + make_vec3( 0.5f*size.x, -0.5f*size.y, 0.f);
  vertices.at(2) = center + make_vec3( 0.5f*size.x, 0.5f*size.y, 0.f);
  vertices.at(3) = center + make_vec3( -0.5f*size.x, 0.5f*size.y, 0.f);
  UUID = context_test.addPatch(center,size);
  prim = context_test.getPrimitivePointer(UUID);
  type = prim->getType();
  center_r = context_test.getPatchPointer(UUID)->getCenter();
  size_r = context_test.getPatchPointer(UUID)->getSize();
  normal_r = prim->getNormal();
  vertices_r = prim->getVertices();
  area_r = prim->getArea();
  color_r = prim->getColor();

  if( type!=PRIMITIVE_TYPE_PATCH ){
    error_count++;
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getType did not return `PRIMITIVE_TYPE_PATCH'." << std::endl;
  }
  if( center_r.x!=center.x || center_r.y!=center.y || center_r.z!=center.z ){
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getCenter returned incorrect value." << std::endl;
  }
  if( size_r.x!=size.x || size_r.y!=size.y ){
    error_count++;
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getSize returned incorrect value." << std::endl;
  }
  if( normal_r.x!=0.f || normal_r.y!=0.f || normal_r.z!=1.f ){
    error_count++;
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getNormal did not return correct value." << std::endl;
  }
  if( vertices.at(0).x!=vertices_r.at(0).x || vertices.at(0).y!=vertices_r.at(0).y || vertices.at(0).z!=vertices_r.at(0).z || 
      vertices.at(1).x!=vertices_r.at(1).x || vertices.at(1).y!=vertices_r.at(1).y || vertices.at(1).z!=vertices_r.at(1).z || 
      vertices.at(2).x!=vertices_r.at(2).x || vertices.at(2).y!=vertices_r.at(2).y || vertices.at(2).z!=vertices_r.at(2).z ||
      vertices.at(3).x!=vertices_r.at(3).x || vertices.at(3).y!=vertices_r.at(3).y || vertices.at(3).z!=vertices_r.at(3).z  ){
    error_count++;
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getVertices did not return correct value." << std::endl;
  }
  if( fabs( area_r - size.x*size.y )>1e-5 ){
    error_count++;
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getArea did not return correct value." << std::endl;
  }
  if( color_r.r!=0.f || color_r.g!=0.f || color_r.b!=0.f ){
    error_count++;
    std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getColor did not return default color of black." << std::endl;
  }

  if( prim->hasTexture() ){
    error_count++;
    std::cerr << "failed: Patch without texture mapping returned true for texture map test." << std::endl;
  }

  //------- Copy Patch --------//

  uint UUID_cpy = context_test.copyPrimitive(UUID);

  vec3 center_cpy = context_test.getPatchPointer(UUID_cpy)->getCenter();
  vec2 size_cpy = context_test.getPatchPointer(UUID_cpy)->getSize();

  if( UUID_cpy != 1 ){
    error_count++;
    std::cerr << "failed: copyPrimitive. Copied patch does not have the correct UUID." << std::endl;
  }
  if( center_cpy.x!=center.x || center_cpy.y!=center.y || center_cpy.z!=center.z  ){
    error_count++;
    std::cerr << "failed: copyPrimitive. Copied patch did not return correct center coordinates." << std::endl;
  }
  if( size_cpy.x!=size.x || size_cpy.y!=size.y ){
    error_count++;
    std::cerr << "failed: copyPrimitive. Copied patch did not return correct size." << std::endl;
  }

  //translate the copied patch
  vec3 shift = make_vec3(5,4,3);
  context_test.getPrimitivePointer(UUID_cpy)->translate(shift);
  center_cpy = context_test.getPatchPointer(UUID_cpy)->getCenter();
  center_r = context_test.getPatchPointer(UUID)->getCenter();

  if( fabs(center_cpy.x-center.x-shift.x)>errtol || fabs(center_cpy.y-center.y-shift.y)>errtol || fabs(center_cpy.z-center.z-shift.z)>errtol || center_r.x!=center.x || center_r.y!=center.y || center_r.z!=center.z ){
    error_count++;
    std::cerr << "failed: copyPrimitive. Copied patch could not be properly translated." << std::endl;
  }

  //------- Delete Patch --------//
  
  context_test.deletePrimitive(UUID);

  if( context_test.getPrimitiveCount()!= 1 ){
    error_count++;
    std::cerr << "failed: deletePrimitive. Patch not properly deleted based on primitive count." << std::endl;
  }
  if( context_test.doesPrimitiveExist(UUID) ){
    error_count++;
    std::cerr << "failed: deletePrimitive. Patch not properly deleted." << std::endl;
  }

  UUID = UUID_cpy;

  //------- Add a Rotated Patch --------//

  center = make_vec3(1,2,3);
  size = make_vec2(1,2);
  rotation.elevation = 0.15*M_PI;
  rotation.azimuth = 0.5f*M_PI;;
  UUID = context_test.addPatch(center,size,rotation);
  prim = context_test.getPrimitivePointer(UUID);
  normal_r = prim->getNormal();

  rotation_r = make_SphericalCoord( 0.5*M_PI-asinf( normal_r.z ), atan2(normal_r.x,normal_r.y) );

  context_test.deletePrimitive(UUID);

  if( fabs(rotation_r.elevation-rotation.elevation)>errtol || fabs(rotation_r.azimuth-rotation.azimuth)>errtol ){
    error_count++;
    std::cerr << "failed: Rotated patch did not return correct normal vector." << std::endl;
  }

  //------- Add Triangle --------//

  vec3 v0,v0_r;
  vec3 v1,v1_r;
  vec3 v2,v2_r;

  //uint addTriangle( const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color );
  v0 = make_vec3(1,2,3);
  v1 = make_vec3(2,4,6);
  v2 = make_vec3(3,6,5);
  vertices.at(0) = v0;
  vertices.at(1) = v1;
  vertices.at(2) = v2;
  color = RGB::red;
  UUID = context_test.addTriangle(v0,v1,v2,color);
  
  prim = context_test.getPrimitivePointer(UUID);
  type = prim->getType();
  normal_r = prim->getNormal();
  vertices_r = prim->getVertices();
  area_r = prim->getArea();
  color_r = prim->getColor();

  if( type!=PRIMITIVE_TYPE_TRIANGLE ){
    error_count++;
    std::cerr << "failed: uint addTriangle(const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color ). Triangle::getType did not return `PRIMITIVE_TYPE_TRIANGLE'." << std::endl;
  }
  normal = cross( v1-v0, v2-v1 );
  normal.normalize();
  if( normal_r.x!=normal.x || normal_r.y!=normal.y || normal_r.z!=normal.z ){
    error_count++;
    std::cerr << "failed: uint addTriangle(const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color ). Triangle::getNormal did not return correct value." << std::endl;
  }
  if( vertices.at(0).x!=vertices_r.at(0).x || vertices.at(0).y!=vertices_r.at(0).y || vertices.at(0).z!=vertices_r.at(0).z || 
      vertices.at(1).x!=vertices_r.at(1).x || vertices.at(1).y!=vertices_r.at(1).y || vertices.at(1).z!=vertices_r.at(1).z || 
      vertices.at(2).x!=vertices_r.at(2).x || vertices.at(2).y!=vertices_r.at(2).y || vertices.at(2).z!=vertices_r.at(2).z ){
     error_count++;
     std::cerr << "failed: uint addTriangle(const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color ). Triangle::getVertices did not return correct value." << std::endl;
  }
  vec3 A(v1-v0);
  vec3 B(v2-v0);
  vec3 C(v2-v1);
  float a = A.magnitude();
  float b = B.magnitude();
  float c = C.magnitude();
  float s = 0.5f*( a+b+c );
  if( area_r!=sqrtf(s*(s-a)*(s-b)*(s-c)) ){
    error_count++;
    std::cerr << "failed: uint addTriangle(const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color ). Triangle::getArea did not return correct value." << std::endl;
  }
  if( color_r.r!=color.r || color_r.g!=color.g || color_r.b!=color.b ){
    error_count++;
    std::cerr << "failed: uint addTriangle(const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color ). Triangle::getColor did not return correct color." << std::endl;
  }

  if( prim->hasTexture() ){
    error_count++;
    std::cerr << "failed: Triangle without texture mapping returned true for texture map test." << std::endl;
  }

  //------- Copy Triangle --------//

  UUID_cpy = context_test.copyPrimitive(UUID);

  std::vector<vec3> vertices_cpy = context_test.getPrimitivePointer(UUID_cpy)->getVertices();

  if( vertices.at(0).x!=vertices_cpy.at(0).x || vertices.at(0).y!=vertices_cpy.at(0).y || vertices.at(0).z!=vertices_cpy.at(0).z || 
      vertices.at(1).x!=vertices_cpy.at(1).x || vertices.at(1).y!=vertices_cpy.at(1).y || vertices.at(1).z!=vertices_cpy.at(1).z || 
      vertices.at(2).x!=vertices_cpy.at(2).x || vertices.at(2).y!=vertices_cpy.at(2).y || vertices.at(2).z!=vertices_cpy.at(2).z ){
     error_count++;
     std::cerr << "failed: copied triangle did not return correct vertices." << std::endl;
  }

  //translate the copied patch
  shift = make_vec3(5,4,3);
  context_test.getPrimitivePointer(UUID_cpy)->translate(shift);

  vertices_cpy = context_test.getPrimitivePointer(UUID_cpy)->getVertices();

  if( vertices.at(0).x!=(vertices_cpy.at(0).x-shift.x) || vertices.at(0).y!=(vertices_cpy.at(0).y-shift.y) || vertices.at(0).z!=(vertices_cpy.at(0).z-shift.z) || 
      vertices.at(1).x!=(vertices_cpy.at(1).x-shift.x) || vertices.at(1).y!=(vertices_cpy.at(1).y-shift.y) || vertices.at(1).z!=(vertices_cpy.at(1).z-shift.z) || 
      vertices.at(2).x!=(vertices_cpy.at(2).x-shift.x) || vertices.at(2).y!=(vertices_cpy.at(2).y-shift.y) || vertices.at(2).z!=(vertices_cpy.at(2).z-shift.z) ){
     error_count++;
     std::cerr << "failed: translated triangle did not return correct vertices." << std::endl;
  }

  //------- Delete Patch --------//
  
  context_test.deletePrimitive(UUID);

  if( context_test.doesPrimitiveExist(UUID) ){
    error_count++;
    std::cerr << "failed: deletePrimitive. Triangle not properly deleted." << std::endl;
  }

  UUID = UUID_cpy;

  //------- Add a Box --------//

  //vector<uint> addBox( const vec3& center, const vec2& size, const int3& subdiv );
  center = make_vec3(1,2,3);
  vec3 size3(3,2,1);
  int3 subdiv(1,1,1);
  UUIDs = context_test.addBox( center, size3, subdiv );
  
  normal_r = context_test.getPrimitivePointer(UUIDs.at(0))->getNormal();
  rotation_r = make_SphericalCoord( 0.5*M_PI-asinf( normal_r.z ), atan2(normal_r.x,normal_r.y) );

  if( fabs(rotation_r.zenith-0.f)>errtol || fabs(rotation_r.azimuth-0.f)>errtol ){
    error_count++;
    std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
  }

  normal_r = context_test.getPrimitivePointer(UUIDs.at(2))->getNormal();
  rotation_r = make_SphericalCoord( 0.5*M_PI-asinf( normal_r.z ), atan2(normal_r.x,normal_r.y) );

  if( fabs(rotation_r.zenith-0.f)>errtol || fabs(rotation_r.azimuth-0.5f*M_PI)>errtol ){
    error_count++;
    std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
  }

  size_r = context_test.getPatchPointer(UUIDs.at(0))->getSize();

  if( fabs(size_r.x-size3.x)>errtol || fabs(size_r.y-size3.z)>errtol ){
    error_count++;
    std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
  }

  size_r = context_test.getPatchPointer(UUIDs.at(2))->getSize();

  if( fabs(size_r.x-size3.y)>errtol || fabs(size_r.y-size3.z)>errtol ){
    error_count++;
    std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
  }

  //------- Add a Rotated Tile --------//

  center = make_vec3(1,2,3);
  size = make_vec2(3,2);
  int2 subdiv2(3,3);
  rotation = make_SphericalCoord( 0.25*M_PI, 1.4*M_PI );
  UUIDs = context_test.addTile( center, size, rotation, subdiv2 );

  for( int p=0; p<UUIDs.size(); p++ ){

      normal_r = context_test.getPrimitivePointer(UUIDs.at(p))->getNormal();

      rotation_r = cart2sphere(normal_r);

      if( fabs(rotation_r.zenith-rotation.zenith)>errtol || fabs(rotation_r.azimuth-rotation.azimuth)>errtol ){
	error_count++;
	std::cerr << "failed: addTile(). Sub-patch normals incorrect." << std::endl;
	break;
      }

  }

  //------- Add a Textured Tile with Transparency (disk) --------//

  center = make_vec3(1,2,3);
  size = make_vec2(3,2);
  subdiv2 = make_int2(5,5);
  rotation = make_SphericalCoord( 0.1*M_PI, 2.4*M_PI );
  UUIDs = context_test.addTile( center, size, rotation, subdiv2, "lib/images/disk_texture.png" );

  float At = 0;
  for( int p=0; p<UUIDs.size(); p++ ){

    float area = context_test.getPrimitivePointer(UUIDs.at(p))->getArea();

    At+=area;

  }

  float area_exact = 0.25*M_PI*size.x*size.y;

  if( fabs(At-area_exact)>0.005 ){
    error_count++;
    std::cerr << "failed: addTile(). Texture masked area is incorrect." << std::endl;
    std::cout << At << " " << area_exact << std::endl;
  }

  //------- Primitive Transformations --------//

  vec2 sz_0(0.5,3.f);
  float A_0 = sz_0.x*sz_0.y;

  float scale = 2.6f;
  
  UUID = context_test.addPatch( make_vec3(0,0,0), sz_0 );
  context_test.getPrimitivePointer(UUID)->scale( make_vec3(scale,scale,scale) );

  float A_1 = context_test.getPrimitivePointer(UUID)->getArea();

  if( fabs( A_1 - scale*scale*A_0 )>1e-5 ){
    error_count ++;
    std::cerr << "failed: Patch scaling - scaled area not correct." << std::endl;
  }

  //------- Primitive Data --------//

  float data = 5;
  context_test.getPrimitivePointer(UUID)->setPrimitiveData("some_data",HELIOS_TYPE_FLOAT,1,&data);

  if( !context_test.getPrimitivePointer(UUID)->doesPrimitiveDataExist("some_data") ){
    error_count ++;
    std::cerr << "failed: setPrimitiveData - data was added but was not actually created." << std::endl;
  }

  float data_return;
  context_test.getPrimitivePointer(UUID)->getPrimitiveData("some_data", data_return);
  if( data_return!=data ){
    error_count ++;
    std::cerr << "failed: set/getPrimitiveData (setting/getting through primitive). Get data did not match set data." << std::endl;
    std::cout << data_return << std::endl;
  }

  context_test.setPrimitiveData(UUID,"some_data",HELIOS_TYPE_FLOAT,1,&data);
  context_test.getPrimitiveData(UUID,"some_data",data_return);
  if( data_return!=data ){
    error_count ++;
    std::cerr << "failed: set/getPrimitiveData (setting/getting through Context). Get data did not match set data." << std::endl;
    std::cout << data_return << std::endl;
  }

  std::vector<float> data_v;
  for( uint i=0; i<5; i++ ){
    data_v.push_back(i);
  }
  context_test.getPrimitivePointer(UUID)->setPrimitiveData("some_data",HELIOS_TYPE_FLOAT,5,&data_v[0] );

  std::vector<float> data_return_v;
  context_test.getPrimitivePointer(UUID)->getPrimitiveData("some_data", data_return_v);
  for( uint i=0; i<5; i++ ){
    if( data_return_v.at(i)!=data_v.at(i) ){
      error_count ++;
      std::cerr << "failed: set/getPrimitiveData (setting/getting through primitive). Get data did not match set data." << std::endl;
      std::cout << data_return << std::endl;
      break;
    }
  }

  data = 10;
  context_test.getPrimitivePointer(UUID)->setPrimitiveData("some_data_2",data);

  if( !context_test.getPrimitivePointer(UUID)->doesPrimitiveDataExist("some_data_2") ){
    error_count ++;
    std::cerr << "failed: setPrimitiveData - data was added but was not actually created." << std::endl;
  }

  context_test.getPrimitivePointer(UUID)->getPrimitiveData("some_data_2", data_return);
  if( data_return!=data ){
    error_count ++;
    std::cerr << "failed: set/getPrimitiveData (setting/getting scalar data). Get data did not match set data." << std::endl;
    std::cout << data_return << std::endl;
  }

  //------- Textures --------- //

  vec2 sizep = make_vec2(2,3);
  
  const char* texture = "lib/images/disk_texture.png";

  vec2 uv0(1,0);
  vec2 uv1(0,0);
  vec2 uv2(0,1);
  vec2 uv3(1,1);

  uint UUIDp = context_test.addPatch( make_vec3(2,3,4), sizep, make_SphericalCoord(0,0), texture, 0.5*(uv3+uv1), uv3-uv1 );

  if( !context_test.getPrimitivePointer(UUIDp)->hasTexture() ){
    error_count ++;
    std::cerr << "failed: Texture-mapped patch was found not to have texture." << std::endl;
  }

  std::string texture2 = context_test.getPrimitivePointer(UUIDp)->getTextureFile();

  if( texture2.compare(texture)!=0 ){
    error_count ++;
    std::cerr << "failed: textures - queried texture file does not match that provided when adding primitive." << std::endl;
  }

  float Ap = context_test.getPrimitivePointer(UUIDp)->getArea();

  if( fabs(Ap-0.25f*M_PI*sizep.x*sizep.y)/(0.25f*M_PI*sizep.x*sizep.y)>0.01f ){
    error_count ++;
    std::cerr << "failed: Texture-masked patch does not have correct area." << std::endl;
  }

  std::vector<vec2> uv;
  uv = context_test.getPrimitivePointer(UUIDp)->getTextureUV();

  if( uv.size()!=4 ){
    error_count ++;
    std::cerr << "failed: Texture (u,v) coordinates for patch should have length of 4." << std::endl;
  }
  if( uv.at(0).x!=uv0.x || uv.at(0).y!=uv0.y || uv.at(1).x!=uv1.x || uv.at(1).y!=uv1.y || uv.at(2).x!=uv2.x || uv.at(2).y!=uv2.y || uv.at(3).x!=uv3.x || uv.at(3).y!=uv3.y ){
    error_count ++;
    std::cerr << "failed: Queried texture (u,v) coordinates do not match that provided when adding primitive." << std::endl;
  }

  uv0 = make_vec2( 0.25, 0.25 );
  uv1 = make_vec2( 0.75, 0.25 );
  uv2 = make_vec2( 0.75, 0.75 );
  uv3 = make_vec2( 0.25, 0.75 );

  uint UUIDp2 = context_test.addPatch( make_vec3(2,3,4), sizep, make_SphericalCoord(0,0), texture, 0.5*(uv2+uv0), uv2-uv0 );

  float area = context_test.getPrimitivePointer(UUIDp2)->getArea();

  if( fabs(area-sizep.x*sizep.y)>0.001f ){
    error_count ++;
    std::cerr << "failed: Patch masked with (u,v) coordinates did not return correct area." << std::endl;
  }

  uv0 = make_vec2( 0, 0 );
  uv1 = make_vec2( 1, 1 );
  uv2 = make_vec2( 0, 1 );

  uint UUIDp3 = context_test.addTriangle( make_vec3(2,3,4), make_vec3(2,3+sizep.y,4), make_vec3(2+sizep.x,3+sizep.y,4), texture, uv0, uv1, uv2 );

  area = context_test.getPrimitivePointer(UUIDp3)->getArea();

  if( fabs(area-0.5*0.25f*M_PI*sizep.x*sizep.y)>0.01f ){
    error_count ++;
    std::cerr << "failed: Triangle masked with (u,v) coordinates did not return correct area." << std::endl;
  }
  
  //------- Global Data --------//

  float gdata = 5;
  context_test.setGlobalData("some_data",HELIOS_TYPE_FLOAT,1,&gdata); 

  if( !context_test.doesGlobalDataExist("some_data") ){
    error_count ++;
    std::cerr << "failed: setGlobalData - data was added but was not actually created." << std::endl;
  }

  float gdata_return;
  context_test.getGlobalData("some_data", gdata_return);
  if( gdata_return!=gdata ){
    error_count ++;
    std::cerr << "failed: set/getGlobalData (setting/getting through primitive). Get data did not match set data." << std::endl;
  }

  std::vector<float> gdata_v;
  for( uint i=0; i<5; i++ ){
    gdata_v.push_back(i);
  }
  context_test.setGlobalData("some_data",HELIOS_TYPE_FLOAT,5,&gdata_v[0] );

  std::vector<float> gdata_return_v;
  context_test.getGlobalData("some_data", gdata_return_v);
  for( uint i=0; i<5; i++ ){
    if( gdata_return_v.at(i)!=gdata_v.at(i) ){
      error_count ++;
      std::cerr << "failed: set/getGlobalData (setting/getting vector data). Get data did not match set data." << std::endl;
      break;
    }
  }

  gdata = 10;
  context_test.setGlobalData("some_data_2",gdata);

  if( !context_test.doesGlobalDataExist("some_data_2") ){
    error_count ++;
    std::cerr << "failed: setGlobalData - data was added but was not actually created." << std::endl;
  }

  context_test.getGlobalData("some_data_2", gdata_return);
  if( gdata_return!=gdata ){
    error_count ++;
    std::cerr << "failed: set/getGlobalData (setting/getting scalar data). Get data did not match set data." << std::endl;
  }



  //------- Cartesian/Spherical Coordinate Conversion --------//
  
  SphericalCoord sph1 = make_SphericalCoord(1.f,0.25*M_PI,1.5*M_PI);
  vec3 cart = sphere2cart(sph1);
  SphericalCoord sph2 = cart2sphere(cart);

  if( fabs(sph1.radius-sph2.radius)>errtol || fabs(sph1.elevation-sph2.elevation)>errtol || fabs(sph1.zenith-sph2.zenith)>errtol || fabs(sph1.azimuth-sph2.azimuth)>errtol ){
    std::cerr << "failed: cart2sphere and sphere2cart are not inverses." << std::endl;
    error_count++;
  }

  //------- Julian Day Conversion ---------//

  int JulianDay = 10;
  int year = 2000; //leap year

  Date date = Julian2Calendar(JulianDay,year);
  if( date.year!=year || date.month!=1 || date.day!=JulianDay ){
    std::cerr << "failed: Julian2Calendar conversion #1 incorrect." << std::endl;
    error_count++;
  }

  JulianDay = 230;
  date = Julian2Calendar(JulianDay,year);
  if( date.year!=year || date.month!=8 || date.day!=17 ){
    std::cerr << "failed: Julian2Calendar conversion #2 incorrect." << std::endl;
    error_count++;
  }

  year = 2001; //non-leap year
  date = Julian2Calendar(JulianDay,year);
  if( date.year!=year || date.month!=8 || date.day!=18 ){
    std::cerr << "failed: Julian2Calendar conversion #3 incorrect." << std::endl;
    error_count++;
  }

  //-------- Timeseries -----------//

  Context context_ts;
  
  Date date_ts( 12, 3, 2010 );
  context_ts.setDate( date_ts );
  
  Time time_ts( 13, 15, 39 );
  context_ts.setTime( time_ts );

  float T0 = 302.3;
  float T1 = 305.3;
  Time time0_ts = time_ts;
  Time time1_ts = make_Time( time_ts.hour, 49, 14 );

  context_ts.addTimeseriesData( "timeseries", T0, date_ts, time0_ts );
  context_ts.addTimeseriesData( "timeseries", T1, date_ts, time1_ts );

  context_ts.setCurrentTimeseriesPoint( "timeseries", 0 );

  if( context_ts.getTimeseriesLength( "timeseries" )!=2 ){
    std::cerr << "failed. getTimeseriesLength() did not give correct length." << std::endl;
    error_count++;
  }

  Date d = context_ts.queryTimeseriesDate( "timeseries", 0 );

  if( d.year!=date_ts.year || d.month!=date_ts.month || d.day!=date_ts.day ){
    std::cerr << "failed. Set/query timeseries date do not match." << std::endl;
    error_count++;
  }

  d = context_ts.queryTimeseriesDate( "timeseries", 1 );

  if( d.year!=date_ts.year || d.month!=date_ts.month || d.day!=date_ts.day ){
    std::cerr << "failed. Set/query timeseries date do not match." << std::endl;
    error_count++;
  }

  Time t = context_ts.queryTimeseriesTime( "timeseries", 0 );

  if( t.hour!=time0_ts.hour || t.minute!=time0_ts.minute || t.second!=time0_ts.second ){
    std::cerr << "failed. Set/query timeseries time do not match." << std::endl;
    error_count++;
  }

  t = context_ts.queryTimeseriesTime( "timeseries", 1 );

  if( t.hour!=time1_ts.hour || t.minute!=time1_ts.minute || t.second!=time1_ts.second ){
    std::cerr << "failed. Set/query timeseries time do not match." << std::endl;
    error_count++;
  }

  float T_ts = context_ts.queryTimeseriesData( "timeseries", 0 );

  if( T_ts!=T0 ){
    std::cerr << "failed: Timeseries set/query data #0 do not match." << std::endl;
    error_count++;
  }

  T_ts = context_ts.queryTimeseriesData( "timeseries", 1 );

  if( T_ts!=T1 ){
    std::cerr << "failed: Timeseries set/query data #1 do not match." << std::endl;
    error_count++;
  }

  T_ts = context_ts.queryTimeseriesData( "timeseries", date_ts, time0_ts );

  if( T_ts!=T0 ){
    std::cerr << "failed: Timeseries set/query data #0 do not match." << std::endl;
    error_count++;
  }

  T_ts = context_ts.queryTimeseriesData( "timeseries", date_ts, time1_ts );

  if( T_ts!=T1 ){
    std::cerr << "failed: Timeseries set/query data #1 do not match." << std::endl;
    error_count++;
  }

  //context_ts.queryTimeseriesData( "timeseries", date_ts, make_Time(0,0,0) ); //this is to test that querying outside of dataset does not throw a segmentation fault.

  context_ts.queryTimeseriesData( "timeseries", date_ts, make_Time(time0_ts.hour,time0_ts.minute,time0_ts.second+10) );

  //------- XML I/O --------//

  Context context_io;

  RGBcolor color_io = RGB::red;

  //date/time

  Date date_io( 12, 3, 2010 );
  context_io.setDate( date_io );
  
  Time time_io( 13, 15, 0 );
  context_io.setTime( time_io );
  
  //patch

  vec3 p_io(2,3,5);
  vec2 size_io(3,2);
  SphericalCoord rot_io(1,0.3*M_PI,0.4*M_PI);

  uint UUIDp_io = context_io.addPatch( p_io, size_io, rot_io, color_io );

  int pdatai_p = 4;
  float pdataf_p = 7.2;
  context_io.setPrimitiveData( UUIDp_io, "pdatai", pdatai_p );
  context_io.setPrimitiveData( UUIDp_io, "pdataf", pdataf_p );

  //triangle

  vec3 v0_io(2,3,5);
  vec3 v1_io(5,1,2);
  vec3 v2_io(8,4,6);
  
  uint UUIDt_io = context_io.addTriangle( v0_io, v1_io, v2_io, color_io );

  int pdatai_t = 9;
  float pdataf_t = 2.7;
  context_io.setPrimitiveData( UUIDt_io, "pdatai", pdatai_t );
  context_io.setPrimitiveData( UUIDt_io, "pdataf", pdataf_t );

  //global data

  std::vector<double> gdatad_io;
  gdatad_io.push_back( 9.432 );
  gdatad_io.push_back( 2939.9292 );
  context_io.setGlobalData( "gdatad", HELIOS_TYPE_DOUBLE, gdatad_io.size(), &gdatad_io[0] );

  //timeseries

  float T0_io = 302.3;
  float T1_io = 305.3;
  Time time0_io = time_io;
  Time time1_io = make_Time( time_io.hour, 30, 0 );

  context_io.addTimeseriesData( "ts_io", T0_io, date_io, time0_io );
  context_io.addTimeseriesData( "ts_io", T1_io, date_io, time1_io );
  
  context_io.writeXML( "xmltest_io.xml" );

  Context context_oi;

  context_oi.loadXML( "xmltest_io.xml" );

  float pdataf;
  int pdatai;

  context_oi.getPrimitiveData(0,"pdataf",pdataf);
  context_oi.getPrimitiveData(0,"pdatai",pdatai);

  if( pdataf!=pdataf_p || pdatai!=pdatai_p ){
    std::cerr << "failed. Patch primitive data write/read do not match." << std::endl;
    error_count ++;
  }

  std::vector<double> gdatad;
  context_oi.getGlobalData("gdatad",gdatad);

  bool failio = false;
  if( gdatad_io.size() != gdatad.size() ){
    failio = true;
  }
  for( int i=0; i<gdatad_io.size(); i++ ){
    if( fabs(gdatad.at(i)-gdatad_io.at(i))>1e-3 ){
      failio = true;
    }
  }
  if( failio ){
    std::cerr << "failed. Global data write/read do not match." << std::endl;
    error_count ++;
  }
  
  //-------------------------------------------//

  if( error_count==0 ){
    std::cout << "passed." << std::endl;
    return 0;
  }else{
    return 1;
  }

}

Texture* Context::addTexture( const char* texture_file ){
  if( textures.find(texture_file)==textures.end() ){//texture has not already been added
    Texture text( texture_file );
    textures[ texture_file ] = text;
  }
  return &textures.at(texture_file);
}

Texture::Texture( const char* texture_file ){
  filename = texture_file;

  //------ determine if transparency channel exists ---------//

  //check if texture file has extension ".png"
  std::string ext;
  for( int i=filename.size()-4; i<filename.size(); i++ ){
    ext.push_back(filename.at(i));
  }
  if( ext.compare(".png")!=0 ){
    hastransparencychannel = false;
  }else{
    hastransparencychannel = PNGHasAlpha( filename.c_str() );
  }
  
  //-------- load transparency channel (if exists) ------------//

  if( hastransparencychannel ){
    transparencydata = readPNGAlpha( filename.c_str() );
  }

  //-------- determine solid fraction --------------//

  if( hastransparencychannel ){
    size_t p = 0.f;
    for( uint j=0; j<transparencydata.size(); j++ ){
      for( uint i=0; i<transparencydata.at(j).size(); i++ ){
	if( transparencydata.at(j).at(i) ){
	  p += 1;
	}
      }
    }
    float sf = float(p)/float(transparencydata.size()*transparencydata.front().size());
    if( sf!=sf ){
      sf = 0.f;
    }
    solidfraction = sf;
  }else{
    solidfraction = 1.f;
  }
  
  
}

std::string Texture::getTextureFile( void ) const{
  return filename;
}

helios::int2 Texture::getSize( void )const {
  return make_int2(transparencydata.front().size(),transparencydata.size());
}

bool Texture::hasTransparencyChannel( void ) const{
  return hastransparencychannel;
}

std::vector<std::vector<bool> >* Texture::getTransparencyData( void ){
  return &transparencydata;
}

float Texture::getSolidFraction( void ) const{
  return solidfraction;
}

void Context::markGeometryClean(void){
  isgeometrydirty = false;
}

void Context::markGeometryDirty(void){
  isgeometrydirty = true;
}

bool Context::isGeometryDirty(void){
  return isgeometrydirty;
}

uint Primitive::getUUID() const{
  return UUID;
}

PrimitiveType Primitive::getType() const{
  return prim_type;
}

void Primitive::getTransformationMatrix( float (&T)[16] )const {
  for( int i=0; i<16; i++ ){
    T[i]=transform[i];
  }
}

void Primitive::setTransformationMatrix( float (&T)[16] ){
  for( int i=0; i<16; i++ ){
    transform[i] = T[i];
  }
}

float Patch::getArea() const{

  vec2 size = getSize();
  
  float area = size.x*size.y*solid_fraction;

  return area;
  
}

float Triangle::getArea() const{

  std::vector<vec3> vertices = getVertices();

  float area = calculateTriangleArea( vertices.at(0), vertices.at(1), vertices.at(2) );
  area = area*solid_fraction;
  
  return area;
}

float Voxel::getArea() const{

  vec3 size(transform[0],transform[5],transform[10]);

  float area = 2.f*size.x*size.y+2*size.x*size.z+2*size.y*size.z;
  
  return area;
}

vec3 Patch::getNormal() const{

  vec3 normal;

  normal.x = transform[2];
  normal.y = transform[6];
  normal.z = transform[10];

  normal.normalize();
  
  return normal;
  
}

vec3 Triangle::getNormal() const{
  std::vector<vec3> vertices = getVertices();
  vec3 norm = cross(vertices.at(1)-vertices.at(0),vertices.at(2)-vertices.at(1));
  norm.normalize();
  return norm;
}

vec3 Voxel::getNormal() const{
  return make_vec3(0,0,0);
}

std::vector<vec3> Patch::getVertices() const{

  vec3 Y[4];
  std::vector<vec3> vertices;
  vertices.resize(4);
  Y[0] = make_vec3( -0.5f, -0.5f, 0.f);
  Y[1] = make_vec3( 0.5f, -0.5f, 0.f);
  Y[2] = make_vec3( 0.5f, 0.5f, 0.f);
  Y[3] = make_vec3( -0.5f, 0.5f, 0.f);

  for( int i=0; i<4; i++ ){
    vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
    vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
    vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
  }
  return vertices;
}

std::vector<vec3> Triangle::getVertices() const{

  vec3 Y[3];
  std::vector<vec3> vertices;
  vertices.resize(4);
  Y[0] = make_vec3( 0.f, 0.f, 0.f);
  Y[1] = make_vec3( 0.f, 1.f, 0.f);
  Y[2] = make_vec3( 1.f, 1.f, 0.f);

  for( int i=0; i<3; i++ ){
    vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
    vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
    vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
  }
  return vertices;
  
}

std::vector<vec3> Voxel::getVertices() const{

  vec3 Y[8];
  std::vector<vec3> vertices;
  vertices.resize(8);
  Y[0] = make_vec3( -0.5f, -0.5f, -0.5f);
  Y[1] = make_vec3( 0.5f, -0.5f, -0.5f);
  Y[2] = make_vec3( 0.5f, 0.5f, -0.5f);
  Y[3] = make_vec3( -0.5f, 0.5f, -0.5f);
  Y[4] = make_vec3( -0.5f, -0.5f, 0.5f);
  Y[5] = make_vec3( 0.5f, -0.5f, 0.5f);
  Y[6] = make_vec3( 0.5f, 0.5f, 0.5f);
  Y[7] = make_vec3( -0.5f, 0.5f, 0.5f);

  for( int i=0; i<8; i++ ){
    vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
    vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
    vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
  }
  return vertices;
}

RGBcolor Primitive::getColor() const{
  return make_RGBcolor(color.r,color.g,color.b);
}

RGBcolor Primitive::getColorRGB() const{
  return make_RGBcolor(color.r,color.g,color.b);
}

RGBAcolor Primitive::getColorRGBA() const{
  return color;
}

void Primitive::setColor( const helios::RGBcolor __color ){
  color = make_RGBAcolor(__color,1);
}

void Primitive::setColor( const helios::RGBAcolor __color ){
  color = __color;
}

Texture* Primitive::getTexture() const{
  return texture;
}

bool Primitive::hasTexture( void ) const{
  if( texture!=NULL ){
    return true;
  }else{
    return false;
  }
}

std::string Primitive::getTextureFile() const{
  if( hasTexture() ){
    return texture->getTextureFile();
  }else{
    std::string blank;
    return blank;
  }
}

std::vector<vec2> Primitive::getTextureUV( void ){
  return uv;
}

void Primitive::overrideTextureColor( void ){
  texturecoloroverridden = true;
}

void Primitive::useTextureColor( void ){
  texturecoloroverridden = false;
}

bool Primitive::isTextureColorOverridden( void ) const{
  return texturecoloroverridden;
}

void Primitive::scale( const vec3 S ){
  float T[16];
  makeScaleMatrix(S,T);
  matmult(T,transform,transform);
}

void Primitive::translate( const helios::vec3 shift ){
  float T[16];
  makeTranslationMatrix(shift,T);
  matmult(T,transform,transform);
}

void Patch::rotate( const float rot, const char* axis ){

  if( strcmp(axis,"z")==0 ){
    float Rz[16];
    makeRotationMatrix(rot,"z",Rz);
    matmult(Rz,transform,transform);
  }else if( strcmp(axis,"y")==0 ){
    float Ry[16];
    makeRotationMatrix(rot,"y",Ry);
    matmult(Ry,transform,transform);
  }else if( strcmp(axis,"x")==0 ){
    float Rx[16];
    makeRotationMatrix(rot,"x",Rx);
    matmult(Rx,transform,transform);
  }else{
    std::cerr << "ERROR (Patch::rotate): Rotation axis should be one of x, y, or z." << std::endl;
    exit(EXIT_FAILURE);
  }

}

void Patch::rotate( const float rot, const helios::vec3 axis ){
  float R[16];
  makeRotationMatrix(rot,axis,R);
  matmult(R,transform,transform);
}

void Triangle::rotate( const float rot, const char* axis ){

  if( strcmp(axis,"z")==0 ){
    float Rz[16];
    makeRotationMatrix(rot,"z",Rz);
    matmult(Rz,transform,transform);
  }else if( strcmp(axis,"y")==0 ){
    float Ry[16];
    makeRotationMatrix(rot,"y",Ry);
    matmult(Ry,transform,transform);
  }else if( strcmp(axis,"x")==0 ){
    float Rx[16];
    makeRotationMatrix(rot,"x",Rx);
    matmult(Rx,transform,transform);
  }else{
    std::cerr << "ERROR (Triangle::rotate): Rotation axis should be one of x, y, or z." << std::endl;
    exit(EXIT_FAILURE);
  }

}

void Triangle::rotate( const float rot, const helios::vec3 axis ){
  float R[16];
  makeRotationMatrix(rot,axis,R);
  matmult(R,transform,transform);
}

void Voxel::rotate( const float rot, const char* axis ){

  float Rz[16];
  makeRotationMatrix(rot,"z",Rz);
  matmult(Rz,transform,transform);
  
}

void Voxel::rotate( const float rot, const helios::vec3 axis ){
  std::cout << "WARNING (Voxel::rotate) - Voxels can only be rotated about the z-axis. Ignoring this call to rotate()." << std::endl;
}

void Triangle::makeTransformationMatrix( const helios::vec3 vertex0, const helios::vec3 vertex1, const helios::vec3 vertex2 ){

  //We need to construct the Affine transformation matrix that transforms some generic triangle to a triangle with vertices at vertex0, vertex1, vertex2.

  //V1 is going to be our generic triangle.  This is the triangle that we'll intersect in the OptiX ray intersection program.  We just need to pass the transformation matrix to OptiX so that we'll end up with the right triangle.

  //We'll assume our generic triangle has vertices
  //v0 = (0,0,0)
  //v1 = (0,1,0)
  //v2 = (1,1,0)
  //this needs to match up with the triangle in triangle_intersect() and triangle_bounds() (see primitiveIntersection.cu).
  //Note that the matrix is padded with 1's to make it 4x4

  float V1[16];

  /* [0,0] */ V1[0] =  0.f;
  /* [0,1] */ V1[1] =  0.f;
  /* [0,2] */ V1[2] =  1.f;

  /* [1,0] */ V1[4] =  0.f;
  /* [1,1] */ V1[5] =  1.f;
  /* [1,2] */ V1[6] =  1.f;

  /* [2,0] */ V1[8] =  0.f;
  /* [2,1] */ V1[9] =  0.f;
  /* [2,2] */ V1[10] = 0.f;

  /* [0,3] */ V1[3] =  1.f;
  /* [1,3] */ V1[7] =  1.f;
  /* [2,3] */ V1[11] = 1.f;
  /* [3,0] */ V1[12] = 1.f;
  /* [3,1] */ V1[13] = 1.f;
  /* [3,2] */ V1[14] = 1.f;
  /* [3,3] */ V1[15] = 1.f;

  //V2 holds the vertex locations we want to transform to
  //Note that the matrix is padded with 1's to make it 4x4

  float V2[16];
   /* [0,0] */ V2[0] =  vertex0.x;
  /* [0,1] */ V2[1] =  vertex1.x;
  /* [0,2] */ V2[2] =  vertex2.x;
  /* [0,3] */ V2[3] =  1.f;
  /* [1,0] */ V2[4] =  vertex0.y;
  /* [1,1] */ V2[5] =  vertex1.y;
  /* [1,2] */ V2[6] =  vertex2.y;
  /* [1,3] */ V2[7] =  1.f;
  /* [2,0] */ V2[8] =  vertex0.z;
  /* [2,1] */ V2[9] =  vertex1.z;
  /* [2,2] */ V2[10] = vertex2.z;
  /* [2,3] */ V2[11] = 1.f;
  /* [3,0] */ V2[12] = 1.f;
  /* [3,1] */ V2[13] = 1.f;
  /* [3,2] */ V2[14] = 1.f;
  /* [3,3] */ V2[15] = 1.f;

  //Now we just need to solve the linear system for our transform matrix T
  // [T][V1] = [V2]  -->
  // [T] = [V2]([V1]^-1)

  double inv[16], det, invV1[16];

  inv[0] = V1[5]  * V1[10] * V1[15] - 
    V1[5]  * V1[11] * V1[14] - 
    V1[9]  * V1[6]  * V1[15] + 
    V1[9]  * V1[7]  * V1[14] +
    V1[13] * V1[6]  * V1[11] - 
    V1[13] * V1[7]  * V1[10];

  inv[4] = -V1[4]  * V1[10] * V1[15] + 
    V1[4]  * V1[11] * V1[14] + 
    V1[8]  * V1[6]  * V1[15] - 
    V1[8]  * V1[7]  * V1[14] - 
    V1[12] * V1[6]  * V1[11] + 
    V1[12] * V1[7]  * V1[10];

  inv[8] = V1[4]  * V1[9] * V1[15] - 
    V1[4]  * V1[11] * V1[13] - 
    V1[8]  * V1[5] * V1[15] + 
    V1[8]  * V1[7] * V1[13] + 
    V1[12] * V1[5] * V1[11] - 
    V1[12] * V1[7] * V1[9];
  
  inv[12] = -V1[4]  * V1[9] * V1[14] + 
    V1[4]  * V1[10] * V1[13] +
    V1[8]  * V1[5] * V1[14] - 
    V1[8]  * V1[6] * V1[13] - 
    V1[12] * V1[5] * V1[10] + 
    V1[12] * V1[6] * V1[9];
  
  inv[1] = -V1[1]  * V1[10] * V1[15] + 
    V1[1]  * V1[11] * V1[14] + 
    V1[9]  * V1[2] * V1[15] - 
    V1[9]  * V1[3] * V1[14] - 
    V1[13] * V1[2] * V1[11] + 
    V1[13] * V1[3] * V1[10];
  
  inv[5] = V1[0]  * V1[10] * V1[15] - 
    V1[0]  * V1[11] * V1[14] - 
    V1[8]  * V1[2] * V1[15] + 
    V1[8]  * V1[3] * V1[14] + 
    V1[12] * V1[2] * V1[11] - 
    V1[12] * V1[3] * V1[10];
  
  inv[9] = -V1[0]  * V1[9] * V1[15] + 
    V1[0]  * V1[11] * V1[13] + 
    V1[8]  * V1[1] * V1[15] - 
    V1[8]  * V1[3] * V1[13] - 
    V1[12] * V1[1] * V1[11] + 
    V1[12] * V1[3] * V1[9];
  
  inv[13] = V1[0]  * V1[9] * V1[14] - 
    V1[0]  * V1[10] * V1[13] - 
    V1[8]  * V1[1] * V1[14] + 
    V1[8]  * V1[2] * V1[13] + 
    V1[12] * V1[1] * V1[10] - 
    V1[12] * V1[2] * V1[9];
  
  inv[2] = V1[1]  * V1[6] * V1[15] - 
    V1[1]  * V1[7] * V1[14] - 
    V1[5]  * V1[2] * V1[15] + 
    V1[5]  * V1[3] * V1[14] + 
    V1[13] * V1[2] * V1[7] - 
    V1[13] * V1[3] * V1[6];
  
  inv[6] = -V1[0]  * V1[6] * V1[15] + 
    V1[0]  * V1[7] * V1[14] + 
    V1[4]  * V1[2] * V1[15] - 
    V1[4]  * V1[3] * V1[14] - 
    V1[12] * V1[2] * V1[7] + 
    V1[12] * V1[3] * V1[6];
  
  inv[10] = V1[0]  * V1[5] * V1[15] - 
    V1[0]  * V1[7] * V1[13] - 
    V1[4]  * V1[1] * V1[15] + 
    V1[4]  * V1[3] * V1[13] + 
    V1[12] * V1[1] * V1[7] - 
    V1[12] * V1[3] * V1[5];
  
  inv[14] = -V1[0]  * V1[5] * V1[14] + 
    V1[0]  * V1[6] * V1[13] + 
    V1[4]  * V1[1] * V1[14] - 
    V1[4]  * V1[2] * V1[13] - 
    V1[12] * V1[1] * V1[6] + 
    V1[12] * V1[2] * V1[5];
  
  inv[3] = -V1[1] * V1[6] * V1[11] + 
    V1[1] * V1[7] * V1[10] + 
    V1[5] * V1[2] * V1[11] - 
    V1[5] * V1[3] * V1[10] - 
    V1[9] * V1[2] * V1[7] + 
    V1[9] * V1[3] * V1[6];
  
  inv[7] = V1[0] * V1[6] * V1[11] - 
    V1[0] * V1[7] * V1[10] - 
    V1[4] * V1[2] * V1[11] + 
    V1[4] * V1[3] * V1[10] + 
    V1[8] * V1[2] * V1[7] - 
    V1[8] * V1[3] * V1[6];
  
  inv[11] = -V1[0] * V1[5] * V1[11] + 
    V1[0] * V1[7] * V1[9] + 
    V1[4] * V1[1] * V1[11] - 
    V1[4] * V1[3] * V1[9] - 
    V1[8] * V1[1] * V1[7] + 
    V1[8] * V1[3] * V1[5];
  
  inv[15] = V1[0] * V1[5] * V1[10] - 
    V1[0] * V1[6] * V1[9] - 
    V1[4] * V1[1] * V1[10] + 
    V1[4] * V1[2] * V1[9] + 
    V1[8] * V1[1] * V1[6] - 
    V1[8] * V1[2] * V1[5];
  
  det = V1[0] * inv[0] + V1[1] * inv[4] + V1[2] * inv[8] + V1[3] * inv[12];
  
  //if (det == 0)
    //return false;
  
  det = 1.0 / det;
  
  for (int i = 0; i < 16; i++)
    invV1[i] = inv[i] * det;
  
  for( int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      transform[ j+i*4 ] = 0.f;
    }
  }

  // Multiply to get transformation matrix [T] = [V2]([V1]^-1)
  for( int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
	transform[ j+i*4 ] += V2[k+i*4]*invV1[j+k*4];
      }
    }
  }
  
}

void Primitive::setPrimitiveData( const char* label, const int& data ){
  std::vector<int> vec;
  vec.push_back(data);
  primitive_data_int[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT;
}

void Primitive::setPrimitiveData( const char* label, const uint& data ){
  std::vector<uint> vec;
  vec.push_back(data);
  primitive_data_uint[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_UINT;
}

void Primitive::setPrimitiveData( const char* label, const float& data ){
  std::vector<float> vec;
  vec.push_back(data);
  primitive_data_float[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_FLOAT;
}

void Primitive::setPrimitiveData( const char* label, const double& data ){
  std::vector<double> vec;
  vec.push_back(data);
  primitive_data_double[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_DOUBLE;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec2& data ){
  std::vector<vec2> vec;
  vec.push_back(data);
  primitive_data_vec2[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_VEC2;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec3& data ){
  std::vector<vec3> vec;
  vec.push_back(data);
  primitive_data_vec3[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_VEC3;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec4& data ){
  std::vector<vec4> vec;
  vec.push_back(data);
  primitive_data_vec4[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_VEC4;
}

void Primitive::setPrimitiveData( const char* label, const helios::int2& data ){
  std::vector<int2> vec;
  vec.push_back(data);
  primitive_data_int2[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT2;
}

void Primitive::setPrimitiveData( const char* label, const helios::int3& data ){
  std::vector<int3> vec;
  vec.push_back(data);
  primitive_data_int3[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT3;
}

void Primitive::setPrimitiveData( const char* label, const helios::int4& data ){
  std::vector<int4> vec;
  vec.push_back(data);
  primitive_data_int4[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_INT4;
}

void Primitive::setPrimitiveData( const char* label, const std::string& data ){
  std::vector<std::string> vec;
  vec.push_back(data);
  primitive_data_string[label] = vec;
  primitive_data_types[label] = HELIOS_TYPE_STRING;
}

void Primitive::setPrimitiveData( const char* label, HeliosDataType type, uint size, void* data ){

  primitive_data_types[label] = type;

  if( type==HELIOS_TYPE_INT ){

    int* data_ptr = (int*)data;

    std::vector<int> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int[label] = vec;

  }else if( type==HELIOS_TYPE_UINT ){

    uint* data_ptr = (uint*)data;

    std::vector<uint> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_uint[label] = vec;

  }else if( type==HELIOS_TYPE_FLOAT ){

    float* data_ptr = (float*)data;

    std::vector<float> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_float[label] = vec;

  }else if( type==HELIOS_TYPE_DOUBLE ){

    double* data_ptr = (double*)data;

    std::vector<double> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_double[label] = vec;

  }else if( type==HELIOS_TYPE_VEC2 ){

    vec2* data_ptr = (vec2*)data;

    std::vector<vec2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_vec2[label] = vec;

  }else if( type==HELIOS_TYPE_VEC3 ){

    vec3* data_ptr = (vec3*)data;

    std::vector<vec3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_vec3[label] = vec;

  }else if( type==HELIOS_TYPE_VEC4 ){

    vec4* data_ptr = (vec4*)data;

    std::vector<vec4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_vec4[label] = vec;

  }else if( type==HELIOS_TYPE_INT2 ){

    int2* data_ptr = (int2*)data;

    std::vector<int2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int2[label] = vec;

  }else if( type==HELIOS_TYPE_INT3 ){

    int3* data_ptr = (int3*)data;

    std::vector<int3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int3[label] = vec;

  }else if( type==HELIOS_TYPE_INT4 ){

    int4* data_ptr = (int4*)data;

    std::vector<int4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_int4[label] = vec;

  }else if( type==HELIOS_TYPE_STRING ){

    std::string* data_ptr = (std::string*)data;

    std::vector<std::string> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    primitive_data_string[label] = vec;
    
  }

}

void Primitive::getPrimitiveData( const char* label, int& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    std::vector<int> d = primitive_data_int.at(label);
    data = d.at(0);
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int, but data '" << label << "' for primitive " << UUID << " does not have type int." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<int>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    std::vector<int> d = primitive_data_int.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int, but data '" << label << "' for primitive " << UUID << " does not have type int." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, uint& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = primitive_data_uint.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type uint, but data '" << label << "' for primitive " << UUID << " does not have type uint." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<uint>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = primitive_data_uint.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type uint, but data '" << label << "' for primitive " << UUID << " does not have type uint." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, float& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = primitive_data_float.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type float, but data '" << label << "' for primitive " << UUID << " does not have type float." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<float>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = primitive_data_float.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type float, but data '" << label << "' for primitive " << UUID << " does not have type float." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, double& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = primitive_data_double.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type double, but data '" << label << "' for primitive " << UUID << " does not have type double." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<double>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = primitive_data_double.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type double, but data '" << label << "' for primitive " << UUID << " does not have type double." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, vec2& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = primitive_data_vec2.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type vec2, but data '" << label << "' for primitive " << UUID << " does not have type vec2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<vec2>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = primitive_data_vec2.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type vec2, but data '" << label << "' for primitive " << UUID << " does not have type vec2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, vec3& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = primitive_data_vec3.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type vec3, but data '" << label << "' for primitive " << UUID << " does not have type vec3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<vec3>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = primitive_data_vec3.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type vec3, but data '" << label << "' for primitive " << UUID << " does not have type vec3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, vec4& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = primitive_data_vec4.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type vec4, but data '" << label << "' for primitive " << UUID << " does not have type vec4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<vec4>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = primitive_data_vec4.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type vec4, but data '" << label << "' for primitive " << UUID << " does not have type vec4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, int2& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = primitive_data_int2.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int2, but data '" << label << "' for primitive " << UUID << " does not have type int2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<int2>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = primitive_data_int2.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int2, but data '" << label << "' for primitive " << UUID << " does not have type int2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, int3& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = primitive_data_int3.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int3, but data '" << label << "' for primitive " << UUID << " does not have type int3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<int3>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = primitive_data_int3.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int3, but data '" << label << "' for primitive " << UUID << " does not have type int3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, int4& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = primitive_data_int4.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int4, but data '" << label << "' for primitive " << UUID << " does not have type int4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<int4>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = primitive_data_int4.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type int4, but data '" << label << "' for primitive " << UUID << " does not have type int4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::string& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = primitive_data_string.at(label);
    data = d.front();
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type string, but data '" << label << "' for primitive " << UUID << " does not have type string." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Primitive::getPrimitiveData( const char* label, std::vector<std::string>& data ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveData): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = primitive_data_string.at(label);
    data = d;
  }else{
    std::cerr << "ERROR (getPrimitiveData): Attempted to get data for type string, but data '" << label << "' for primitive " << UUID << " does not have type string." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

HeliosDataType Primitive::getPrimitiveDataType( const char* label ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveDataType): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  return primitive_data_types.at(label);

}

uint Primitive::getPrimitiveDataSize( const char* label ) const{

  if( !doesPrimitiveDataExist( label ) ){
    std::cerr << "ERROR (getPrimitiveDataSize): Primitive data " << label << " does not exist for primitive " << UUID << std::endl;
    exit(EXIT_FAILURE);
  }

  HeliosDataType type = primitive_data_types.at(label);

  if( type==HELIOS_TYPE_INT ){
    return primitive_data_int.at(label).size();
  }else if( type==HELIOS_TYPE_UINT ){
    return primitive_data_uint.at(label).size();
  }else if( type==HELIOS_TYPE_FLOAT ){
    return primitive_data_float.at(label).size();
  }else if( type==HELIOS_TYPE_DOUBLE ){
    return primitive_data_double.at(label).size();
  }else if( type==HELIOS_TYPE_VEC2 ){
    return primitive_data_vec2.at(label).size();
  }else if( type==HELIOS_TYPE_VEC3 ){
    return primitive_data_vec3.at(label).size();
  }else if( type==HELIOS_TYPE_VEC4 ){
    return primitive_data_vec4.at(label).size();
  }else if( type==HELIOS_TYPE_INT2 ){
    return primitive_data_int2.at(label).size();
  }else if( type==HELIOS_TYPE_INT3 ){
    return primitive_data_int3.at(label).size();
  }else if( type==HELIOS_TYPE_INT4 ){
    return primitive_data_int4.at(label).size();
  }else if( type==HELIOS_TYPE_STRING ){
    return primitive_data_string.at(label).size();
  }else{
    assert( false );
  }

  return 0;

}

bool Primitive::doesPrimitiveDataExist( const char* label ) const{

  if( primitive_data_types.find(label) == primitive_data_types.end() ){
    return false;
  }else{
    return true;
  }

}

std::vector<std::string> Primitive::listPrimitiveData( void ) const{

  std::vector<std::string> labels;
  
  for( std::map<std::string,HeliosDataType>::const_iterator iter = primitive_data_types.begin(); iter!=primitive_data_types.end(); ++iter ){
    labels.push_back( iter->first );
  }

  return labels;
  
}
  
void Context::setPrimitiveData( const uint UUID, const char* label, const int& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const uint& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const float& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const double& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const helios::vec2& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const helios::vec3& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const helios::vec4& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const helios::int2& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const helios::int3& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const helios::int4& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, const std::string& data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint UUID, const char* label, HeliosDataType type, uint size, void* data ){
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (setPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->setPrimitiveData(label,type,size,data);
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const int& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const uint& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const float& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const double& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::vec2& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::vec3& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::vec4& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::int2& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::int3& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const helios::int4& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::setPrimitiveData( const std::vector<uint> UUIDs, const char* label, const std::string& data ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    setPrimitiveData( UUIDs.at(p), label, data );
  }
}

void Context::getPrimitiveData( const uint UUID, const char* label, int& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, uint& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<uint>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, float& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<float>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, double& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<double>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec2& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec2>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec3& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec3>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec4& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec4>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int2& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int2>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int3& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int3>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int4& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int4>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::string& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<std::string>& data ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveData): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  primitives.at(UUID)->getPrimitiveData(label,data);
}

HeliosDataType Context::getPrimitiveDataType( const uint UUID, const char* label )const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveDataType): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  return primitives.at(UUID)->getPrimitiveDataType(label);
}

uint Context::getPrimitiveDataSize( const uint UUID, const char* label )const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitiveDataSize): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  return primitives.at(UUID)->getPrimitiveDataSize(label);
}

bool Context::doesPrimitiveDataExist( const uint UUID, const char* label ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (doesPrimitiveDataExist): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  return primitives.at(UUID)->doesPrimitiveDataExist(label);
}

void Context::setGlobalData( const char* label, const int& data ){
  std::vector<int> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_INT;
  globaldata[label].size = 1;
  globaldata[label].global_data_int = vec;
}

void Context::setGlobalData( const char* label, const uint& data ){
  std::vector<uint> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_UINT;
  globaldata[label].size = 1;
  globaldata[label].global_data_uint = vec;
}

void Context::setGlobalData( const char* label, const float& data ){
  std::vector<float> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_FLOAT;
  globaldata[label].size = 1;
  globaldata[label].global_data_float = vec;
}

void Context::setGlobalData( const char* label, const double& data ){
  std::vector<double> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_DOUBLE;
  globaldata[label].size = 1;
  globaldata[label].global_data_double = vec;
}

void Context::setGlobalData( const char* label, const helios::vec2& data ){
  std::vector<vec2> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_VEC2;
  globaldata[label].size = 1;
  globaldata[label].global_data_vec2 = vec;
}

void Context::setGlobalData( const char* label, const helios::vec3& data ){
  std::vector<vec3> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_VEC3;
  globaldata[label].size = 1;
  globaldata[label].global_data_vec3 = vec;
}

void Context::setGlobalData( const char* label, const helios::vec4& data ){
  std::vector<vec4> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_VEC4;
  globaldata[label].size = 1;
  globaldata[label].global_data_vec4 = vec;
}

void Context::setGlobalData( const char* label, const helios::int2& data ){
  std::vector<int2> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_INT2;
  globaldata[label].size = 1;
  globaldata[label].global_data_int2 = vec;
}

void Context::setGlobalData( const char* label, const helios::int3& data ){
  std::vector<int3> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_INT3;
  globaldata[label].size = 1;
  globaldata[label].global_data_int3 = vec;
}

void Context::setGlobalData( const char* label, const helios::int4& data ){
  std::vector<int4> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_INT4;
  globaldata[label].size = 1;
  globaldata[label].global_data_int4 = vec;
}

void Context::setGlobalData( const char* label, const std::string& data ){
  std::vector<std::string> vec;
  vec.push_back(data);
  globaldata[label].type = HELIOS_TYPE_STRING;
  globaldata[label].size = 1;
  globaldata[label].global_data_string = vec;
}

void Context::setGlobalData( const char* label, HeliosDataType type, size_t size, void* data ){

  globaldata[label].type = type;
  globaldata[label].size = size;

  if( type==HELIOS_TYPE_INT ){

    int* data_ptr = (int*)data;

    std::vector<int> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int = vec;

  }else if( type==HELIOS_TYPE_UINT ){

    uint* data_ptr = (uint*)data;

    std::vector<uint> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_uint = vec;

  }else if( type==HELIOS_TYPE_FLOAT ){

    float* data_ptr = (float*)data;

    std::vector<float> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_float = vec;

  }else if( type==HELIOS_TYPE_DOUBLE ){

    double* data_ptr = (double*)data;

    std::vector<double> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_double = vec;

  }else if( type==HELIOS_TYPE_VEC2 ){

    vec2* data_ptr = (vec2*)data;

    std::vector<vec2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_vec2 = vec;

  }else if( type==HELIOS_TYPE_VEC3 ){

    vec3* data_ptr = (vec3*)data;

    std::vector<vec3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_vec3= vec;

  }else if( type==HELIOS_TYPE_VEC4 ){

    vec4* data_ptr = (vec4*)data;

    std::vector<vec4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_vec4 = vec;

  }else if( type==HELIOS_TYPE_INT2 ){

    int2* data_ptr = (int2*)data;

    std::vector<int2> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int2 = vec;

  }else if( type==HELIOS_TYPE_INT3 ){

    int3* data_ptr = (int3*)data;

    std::vector<int3> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int3 = vec;

  }else if( type==HELIOS_TYPE_INT4 ){

    int4* data_ptr = (int4*)data;

    std::vector<int4> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_int4 = vec;

  }else if( type==HELIOS_TYPE_STRING ){

    std::string* data_ptr = (std::string*)data;

    std::vector<std::string> vec;
    vec.resize(size);
    for( size_t i=0; i<size; i++ ){
      vec.at(i) = data_ptr[i];
    }
    globaldata[label].global_data_string = vec;
    
  }

}

void Context::getGlobalData( const char* label, int& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT ){
    std::vector<int> d = gdata.global_data_int;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type int, but data does not have type int." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<int>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT ){
    std::vector<int> d = gdata.global_data_int;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type int, but data  does not have type int." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, uint& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = gdata.global_data_uint;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type uint, but data does not have type uint." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<uint>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_UINT ){
    std::vector<uint> d = gdata.global_data_uint;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type uint, but data does not have type int." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, float& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = gdata.global_data_float;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type float, but data does not have type float." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<float>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_FLOAT ){
    std::vector<float> d = gdata.global_data_float;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type float, but data does not have type float." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, double& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = gdata.global_data_double;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type double, but data does not have type double." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<double>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_DOUBLE ){
    std::vector<double> d = gdata.global_data_double;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type double, but data does not have type double." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, helios::vec2& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = gdata.global_data_vec2;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type vec2, but data does not have type vec2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<helios::vec2>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC2 ){
    std::vector<vec2> d = gdata.global_data_vec2;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type vec2, but data does not have type vec2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, helios::vec3& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = gdata.global_data_vec3;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type vec3, but data does not have type vec3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<helios::vec3>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC3 ){
    std::vector<vec3> d = gdata.global_data_vec3;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type vec3, but data does not have type vec3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, helios::vec4& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = gdata.global_data_vec4;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type vec4, but data does not have type vec4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<helios::vec4>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_VEC4 ){
    std::vector<vec4> d = gdata.global_data_vec4;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type vec4, but data does not have type vec4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, helios::int2& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = gdata.global_data_int2;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type int2, but data does not have type int2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<helios::int2>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT2 ){
    std::vector<int2> d = gdata.global_data_int2;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type int2, but data does not have type int2." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, helios::int3& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = gdata.global_data_int3;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type int3, but data does not have type int3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<helios::int3>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT3 ){
    std::vector<int3> d = gdata.global_data_int3;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type int3, but data does not have type int3." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, helios::int4& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = gdata.global_data_int4;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type int4, but data does not have type int4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<helios::int4>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_INT4 ){
    std::vector<int4> d = gdata.global_data_int4;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type int4, but data does not have type int4." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::string& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Primitive data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = gdata.global_data_string;
    data = d.front();
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get global data for type string, but data does not have type string." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

void Context::getGlobalData( const char* label, std::vector<std::string>& data ) const{

  if( !doesGlobalDataExist( label ) ){
    std::cerr << "ERROR (getGlobalData): Global data " << label << " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  GlobalData gdata = globaldata.at(label);

  if( gdata.type==HELIOS_TYPE_STRING ){
    std::vector<std::string> d = gdata.global_data_string;
    data = d;
  }else{
    std::cerr << "ERROR (getGlobalData): Attempted to get data for type string, but data does not have type string." << std::endl;
    exit(EXIT_FAILURE);
  }
  
}

bool Context::doesGlobalDataExist( const char* label ) const{

  if( globaldata.find(label) == globaldata.end() ){
    return false;
  }else{
    return true;
  }

}


Patch::Patch( const RGBAcolor _color_, const uint _UUID_ ){

  makeIdentityMatrix( transform );
   
  color = _color_;
  assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
  UUID = _UUID_;
  prim_type = PRIMITIVE_TYPE_PATCH;
  solid_fraction = 1.f;
  texture = 0;
  texturecoloroverridden = false;

}

Patch::Patch( Texture* _texture_, const uint _UUID_ ){

  makeIdentityMatrix( transform );

  UUID = _UUID_;
  prim_type = PRIMITIVE_TYPE_PATCH;
  texture = _texture_;
  solid_fraction = texture->getSolidFraction();
  texturecoloroverridden = false;
  
}

Patch::Patch( Texture* _texture_, const std::vector<vec2> _uv_, const float _solid_fraction_, const uint _UUID_ ){

  makeIdentityMatrix( transform );
  
  UUID = _UUID_;
  prim_type = PRIMITIVE_TYPE_PATCH;

  texture = _texture_;
  uv = _uv_;
  solid_fraction = _solid_fraction_;
  texturecoloroverridden = false;

}

helios::vec2 Patch::getSize(void) const{
  std::vector<vec3> vertices = getVertices();
  float l = (vertices.at(1)-vertices.at(0)).magnitude();
  float w = (vertices.at(3)-vertices.at(0)).magnitude();
  return make_vec2(l,w);
}

helios::vec3 Patch::getCenter(void) const{
  return make_vec3(transform[3],transform[7],transform[11]);
}

Triangle::Triangle(  const vec3 vertex0, const vec3 vertex1, const vec3 vertex2, const RGBAcolor _color_, const uint _UUID_ ){

  makeTransformationMatrix(vertex0,vertex1,vertex2);
    color = _color_;
  UUID = _UUID_;
  prim_type = PRIMITIVE_TYPE_TRIANGLE;
  texture = 0;
  solid_fraction = 1.f;
  texturecoloroverridden = false;

}

Triangle::Triangle( const vec3 vertex0, const vec3 vertex1, const vec3 vertex2, Texture* _texture_, const std::vector<vec2> _uv_, const float _solid_fraction_, const uint _UUID_ ){

  makeTransformationMatrix(vertex0,vertex1,vertex2);
  color = make_RGBAcolor(RGB::red,1);
  UUID = _UUID_;
  prim_type = PRIMITIVE_TYPE_TRIANGLE;

  texture = _texture_;
  uv = _uv_;
  solid_fraction = _solid_fraction_;
  texturecoloroverridden = false;

}

vec3 Triangle::getVertex( int number ){

  if( number<0 || number>2 ){
    std::cerr << "getVertex ERROR: vertex number must be 0, 1, or 2." << std::endl;
    exit(EXIT_FAILURE);
  }

  vec3 Y[3];
  Y[0] = make_vec3( 0.f, 0.f, 0.f);
  Y[1] = make_vec3( 0.f, 1.f, 0.f);
  Y[2] = make_vec3( 1.f, 1.f, 0.f);

  vec3 vertex;
  
  vertex.x = transform[0] * Y[number].x + transform[1] * Y[number].y + transform[2] * Y[number].z + transform[3];
  vertex.y = transform[4] * Y[number].x + transform[5] * Y[number].y + transform[6] * Y[number].z + transform[7];
  vertex.z = transform[8] * Y[number].x + transform[9] * Y[number].y + transform[10] * Y[number].z + transform[11];

  return vertex;
  
}

Voxel::Voxel( const RGBAcolor _color_, const uint _UUID_ ){

  makeIdentityMatrix(transform);
  
  color = _color_;
  assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
  UUID = _UUID_;
  prim_type = PRIMITIVE_TYPE_VOXEL;
  texture = 0;
  texturecoloroverridden = false;
  
}

float Voxel::getVolume(void){
  
  return transform[0]*transform[5]*transform[10];
}

vec3 Voxel::getCenter(void){
  return make_vec3(transform[3],transform[7],transform[11]);
}

vec3 Voxel::getSize(void){
  return make_vec3(transform[0],transform[5],transform[10]);
}

void Context::setDate( int day, int month, int year ){

  if( day<1 || day>31 ){
    std::cerr << "ERROR (setDate): Day of month is out of range (day of " << day << " was given)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( month<1 || month>12){
    std::cerr << "ERROR (setDate): Month of year is out of range (month of " << month << " was given)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( year<1000 ){
    std::cerr << "ERROR (setDate): Year should be specified in YYYY format." << std::endl;
    exit(EXIT_FAILURE);
  }

  sim_date = make_Date(day,month,year);

  return;

}

void Context::setDate( Date date ){

  if( date.day<1 || date.day>31 ){
    std::cerr << "ERROR (setDate): Day of month is out of range (day of " << date.day << " was given)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( date.month<1 || date.month>12){
    std::cerr << "ERROR (setDate): Month of year is out of range (month of " << date.month << " was given)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( date.year<1000 ){
    std::cerr << "ERROR (setDate): Year should be specified in YYYY format." << std::endl;
    exit(EXIT_FAILURE);
  }

  sim_date = date;

  return;

}

void Context::setDate( int Julian_day, int year ){

  if( Julian_day<1 || Julian_day>366 ){
    std::cerr << "ERROR (setDate): Julian day out of range." << std::endl;
    exit(EXIT_FAILURE);
  }else if( year<1000 ){
    std::cerr << "ERROR (setDate): Year should be specified in YYYY format." << std::endl;
    exit(EXIT_FAILURE);
  }

  sim_date = CalendarDay( Julian_day, year );

  return;
  
}

Date Context::getDate( void ) const{
  return sim_date;
}

const char* Context::getMonthString( void ) const{
  if( sim_date.month==1 ){
    return "JAN";
  }else if( sim_date.month==2 ){
    return "FEB";
  }else if( sim_date.month==3 ){
    return "MAR";
  }else if( sim_date.month==4 ){
    return "APR";
  }else if( sim_date.month==5 ){
    return "MAY";
  }else if( sim_date.month==6 ){
    return "JUN";
  }else if( sim_date.month==7 ){
    return "JUL";
  }else if( sim_date.month==8 ){
    return "AUG";
  }else if( sim_date.month==9 ){
    return "SEP";
  }else if( sim_date.month==10 ){
    return "OCT";
  }else if( sim_date.month==11 ){
    return "NOV";
  }else{
    return "DEC";
  }

}

int Context::getJulianDate( void ) const{
  return JulianDay( sim_date.day, sim_date.month, sim_date.year );
}

void Context::setTime( int minute, int hour ){
  setTime(0,minute,hour);
}
  
void Context::setTime( int second, int minute, int hour ){

  if( second<0 || second>59 ){
    std::cerr << "ERROR (setTime): Second out of range (0-59)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( minute<0 || minute>59 ){
    std::cerr << "ERROR (setTime): Minute out of range (0-59)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( hour<0 || hour>23 ){
    std::cerr << "ERROR (setTime): Hour out of range (0-23)." << std::endl;
    exit(EXIT_FAILURE);
  }

  sim_time = make_Time(hour,minute,second);
  
}

void Context::setTime( Time time ){

  if( time.minute<0 || time.minute>59 ){
    std::cerr << "ERROR (setTime): Minute out of range (0-59)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( time.hour<0 || time.hour>23 ){
    std::cerr << "ERROR (setTime): Hour out of range (0-23)." << std::endl;
    exit(EXIT_FAILURE);
  }

  sim_time = time;
  
}

Time Context::getTime( void ) const{
  return sim_time;
}

float Context::randu(void){
  return unif_distribution(generator);
}

float Context::randu( float min, float max ){
  if( max<min ){
    std::cerr << "ERROR (randu): Maximum value of range must be greater than minimum value of range." << std::endl;
    exit(EXIT_FAILURE);
  }else if( max==min ){
    return min;
  }else{
    return min+unif_distribution(generator)*(max-min);
  }
}

int Context::randu( int min, int max ){
  if( max<min ){
    std::cerr << "ERROR (randu): Maximum value of range must be greater than minimum value of range." << std::endl;
    exit(EXIT_FAILURE);
  }else if( max==min ){
    return min;
  }else{
    return min+round(unif_distribution(generator)*float(max-min));
  }
}

float Context::randn(void){
  return norm_distribution(generator);
}

float Context::randn( float mean, float stddev ){
  return mean+norm_distribution(generator)*fabs(stddev);
}
  
uint Context::addPatch( const vec3& center, const vec2& size ){
  return addPatch(center,size,make_SphericalCoord(0,0),make_RGBAcolor(0,0,0,1));
}

uint Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation ){
  return addPatch(center,size,rotation,make_RGBAcolor(0,0,0,1));
}

uint Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation, const RGBcolor& color ){
  return addPatch(center,size,rotation,make_RGBAcolor(color,1));
}

uint Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation, const RGBAcolor& color ){

  Patch* patch_new = (new Patch( color, currentUUID ));

  assert( size.x>0.f && size.y>0.f );
  patch_new->scale( make_vec3(size.x,size.y,1) );
  
  if( rotation.elevation!=0 ){
    patch_new->rotate(-rotation.elevation, "x");
  }
  if( rotation.azimuth!=0 ){
    patch_new->rotate(-rotation.azimuth, "z");
  }

  patch_new->translate( center );
  
  primitives[currentUUID] = patch_new;
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1;
}

uint Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation, const char* texture_file ){

  Texture* texture = addTexture( texture_file );

  Patch* patch_new = (new Patch( texture, currentUUID ));

  assert( size.x>0.f && size.y>0.f );
  patch_new->scale( make_vec3(size.x,size.y,1) );
  
  if( rotation.elevation!=0 ){
    patch_new->rotate(-rotation.elevation, "x");
  }
  if( rotation.azimuth!=0 ){
    patch_new->rotate(-rotation.azimuth, "z");
  }

  patch_new->translate( center );
  
  primitives[currentUUID] = patch_new;
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1;
}

uint Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation,const char* texture_file, const helios::vec2& uv_center, const helios::vec2& uv_size ){

  if( uv_center.x-0.5*uv_size.x<-1e-3 || uv_center.y-0.5*uv_size.y<-1e-3 || uv_center.x+0.5*uv_size.x-1.f>1e-3 || uv_center.y+0.5*uv_size.y-1.f>1e-3 ){
    std::cerr << "ERROR (addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1." << std::endl;
    std::cout << (uv_center.x-0.5*uv_size.x>=0.f) << " " << (uv_center.y-0.5*uv_size.y>=0.f) << " " << (uv_center.x+0.5*uv_size.x<=1.f) << " " << (uv_center.y+0.5*uv_size.y<=1.f) << std::endl;
    exit(EXIT_FAILURE);
  }

  Texture* texture = addTexture( texture_file );

  std::vector<helios::vec2> uv;
  uv.resize(4);
  uv.at(0) = uv_center+make_vec2(+0.5*uv_size.x,-0.5*uv_size.y);
  uv.at(1) = uv_center+make_vec2(-0.5*uv_size.x,-0.5*uv_size.y);
  uv.at(2) =  uv_center+make_vec2(-0.5*uv_size.x,+0.5*uv_size.y);
  uv.at(3) =  uv_center+make_vec2(+0.5*uv_size.x,+0.5*uv_size.y);

  float solid_fraction;
  if( texture->hasTransparencyChannel() ){
    std::vector<std::vector<bool> >* alpha = texture->getTransparencyData();
    int A = 0;
    int At = 0;
    int2 sz = texture->getSize();
    int2 uv_min( floor(uv.at(1).x*sz.x), floor(uv.at(1).y*sz.y) );
    int2 uv_max( floor(uv.at(3).x*sz.x), floor(uv.at(3).y*sz.y) );
    for( int j=uv_min.y; j<uv_max.y; j++ ){
      for( int i=uv_min.x; i<uv_max.x; i++ ){
	At += 1;
	if( alpha->at(j).at(i) ){
	  A += 1;
	}
      }
    }
    if( At==0 ){
      solid_fraction = 0;
    }else{
      solid_fraction = float(A)/float(At);
    }
  }else{
    solid_fraction = 1.f;
  }
  
  Patch* patch_new = (new Patch( texture, uv, solid_fraction, currentUUID ));

  assert( size.x>0.f && size.y>0.f );
  patch_new->scale( make_vec3(size.x,size.y,1) );
  
  if( rotation.elevation!=0 ){
    patch_new->rotate(-rotation.elevation, "x");
  }
  if( rotation.azimuth!=0 ){
    patch_new->rotate(-rotation.azimuth, "z");
  }

  patch_new->translate( center );
  
  primitives[currentUUID] = patch_new;
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1;
}

uint Context::addTriangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2 ){
  return addTriangle( vertex0, vertex1, vertex2, make_RGBAcolor(0,0,0,1) );
}

uint Context::addTriangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2, const RGBcolor& color ){
  return addTriangle( vertex0, vertex1, vertex2, make_RGBAcolor(color,1) );
}

bool edgeFunction(const helios::vec2 &a, const helios::vec2 &b, const helios::vec2 &c) 
{ 
    return ((c.y - a.y) * (b.x - a.x)-(c.x - a.x) * (b.y - a.y) >= 0); 
} 

uint Context::addTriangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2, const RGBAcolor& color ){

  Triangle* tri_new = (new Triangle( vertex0, vertex1, vertex2, color, currentUUID ));
  primitives[currentUUID] = tri_new;
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1;  
}

uint Context::addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texture_file, const helios::vec2& uv0, const helios::vec2& uv1, const helios::vec2& uv2 ){

  Texture* texture = addTexture( texture_file );

  std::vector<helios::vec2> uv;
  uv.resize(3);
  uv.at(0) = uv0;
  uv.at(1) = uv1;
  uv.at(2) = uv2;

  float solid_fraction;
  if( texture->hasTransparencyChannel() ){
    std::vector<std::vector<bool> >* alpha = texture->getTransparencyData();
    int2 sz = texture->getSize();
    int2 uv_min( round(fmin(fmin(uv0.x,uv1.x),uv2.x)*sz.x), round(fmin(fmin(uv0.y,uv1.y),uv2.y)*sz.y) );
    int2 uv_max( round(fmax(fmax(uv0.x,uv1.x),uv2.x)*sz.x), round(fmax(fmax(uv0.y,uv1.y),uv2.y)*sz.y) );
    int A = 0;
    int At = 0;
    vec2 xy;
    for( int j=uv_min.y; j<uv_max.y; j++ ){
      for( int i=uv_min.x; i<uv_max.x; i++ ){
	xy.x = float(i+0.5)/float(sz.x-1);
	xy.y = float(j+0.5)/float(sz.y-1);
	if( edgeFunction( uv.at(0), uv.at(1), xy ) && edgeFunction( uv.at(1), uv.at(2), xy )&& edgeFunction( uv.at(2), uv.at(0), xy ) ){
	  At += 1;
	  if( alpha->at(j).at(i) ){
	    A += 1;
	  }
	}
      }
    }
    if( At==0 ){
      solid_fraction = 0;
    }else{
      solid_fraction = float(A)/float(At);
    }
  }else{
    solid_fraction = 1.f;
  }

  Triangle* tri_new = (new Triangle( vertex0, vertex1, vertex2, texture, uv, solid_fraction, currentUUID ));
  primitives[currentUUID] = tri_new;
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1; 
}

uint Context::addVoxel( const vec3& center, const vec3& size ){
  return addVoxel(center,size,0,make_RGBAcolor(0,0,0,1));
}

uint Context::addVoxel( const vec3& center, const vec3& size, const float& rotation ){
  return addVoxel(center,size,rotation,make_RGBAcolor(0,0,0,1));
}

uint Context::addVoxel( const vec3& center, const vec3& size, const float& rotation, const RGBcolor& color ){
  return addVoxel(center,size,rotation,make_RGBAcolor(color,1));
}

uint Context::addVoxel( const vec3& center, const vec3& size, const float& rotation, const RGBAcolor& color ){

  Voxel* voxel_new = (new Voxel( color, currentUUID ));

  voxel_new->scale( size );
  
  if( rotation!=0 ){
    voxel_new->rotate( rotation, "z" );
  }

  voxel_new->translate( center );
  
  primitives[currentUUID] = voxel_new;
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1;
}

void Context::translatePrimitive( const uint UUID, const vec3 shift ){
  getPrimitivePointer(UUID)->translate(shift);
}

void Context::translatePrimitive( const std::vector<uint> UUIDs, const vec3 shift ){
  for( uint p=0; p<UUIDs.size(); p++ ){
    getPrimitivePointer(UUIDs.at(p))->translate(shift);
  }
}

void Context::rotatePrimitive( const uint UUID, const float rot, const char* axis ){
  getPrimitivePointer(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive( const std::vector<uint> UUIDs, const float rot, const char* axis ){
  for( uint p=0; p<UUIDs.size(); p++ ){
    getPrimitivePointer(UUIDs.at(p))->rotate(rot,axis);
  }
}

void Context::rotatePrimitive( const uint UUID, const float rot, const helios::vec3 axis ){
  getPrimitivePointer(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive( const std::vector<uint> UUIDs, const float rot, helios::vec3 axis ){
  for( uint p=0; p<UUIDs.size(); p++ ){
    getPrimitivePointer(UUIDs.at(p))->rotate(rot,axis);
  }
}

void Context::scalePrimitive( const uint UUID, const helios::vec3 S ){
  getPrimitivePointer(UUID)->scale(S);
}

void Context::scalePrimitive( const std::vector<uint> UUIDs, const helios::vec3 S ){
  for( uint p=0; p<UUIDs.size(); p++ ){
    getPrimitivePointer(UUIDs.at(p))->scale(S);
  }
}

void Context::deletePrimitive( const std::vector<uint> UUIDs ){
  for( size_t p=0; p<UUIDs.size(); p++ ){
    deletePrimitive( UUIDs.at(p) );
  }
}

void Context::deletePrimitive( const uint UUID ){

  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (deletePrimitive: UUID of " << UUID << " not found in the context." << std::endl;
    exit(EXIT_FAILURE);
  }
  
  Primitive* prim = primitives.at(UUID);
  delete prim;
  primitives.erase(UUID);

  markGeometryDirty();
  
}

std::vector<uint> Context::copyPrimitive( const std::vector<uint> UUIDs ){

  std::vector<uint> UUIDs_copy;
  for( size_t p=0; p<UUIDs.size(); p++ ){
    UUIDs_copy.push_back( copyPrimitive( UUIDs.at(p) ) );
  }

  return UUIDs_copy;
  
}

uint Context::copyPrimitive( const uint UUID ){

  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (copyPrimitive: UUID of " << UUID << " not found in the context." << std::endl;
    exit(EXIT_FAILURE);
  }

  PrimitiveType type = primitives.at(UUID)->getType();

  if( type==PRIMITIVE_TYPE_PATCH ){
    Patch* p = getPatchPointer(UUID);
    std::vector<vec2> uv = p->getTextureUV();
    vec2 size = p->getSize();
    float solid_fraction = p->getArea()/(size.x*size.y);
    Patch* patch_new;
    if( !p->hasTexture() ){
      patch_new = (new Patch( p->getColorRGBA(), currentUUID ));
    }else{
      Texture* texture = p->getTexture();
      if( uv.size()==4 ){
	patch_new = (new Patch( texture, uv, solid_fraction, currentUUID ));
      }else{
	patch_new = (new Patch( texture, currentUUID ));
      }
    }
    float transform[16];
    p->getTransformationMatrix(transform);
    patch_new->setTransformationMatrix(transform);
    primitives[currentUUID] = patch_new;
  }else if( type==PRIMITIVE_TYPE_TRIANGLE ){
    Triangle* p = getTrianglePointer(UUID);
    std::vector<vec3> vertices = p->getVertices();
    std::vector<vec2> uv = p->getTextureUV();
    Triangle* tri_new;
    if( !p->hasTexture() ){
      tri_new = (new Triangle( vertices.at(0), vertices.at(1), vertices.at(2), p->getColorRGBA(), currentUUID ));
    }else{
      Texture* texture = p->getTexture();
      float solid_fraction = p->getArea()/calculateTriangleArea( vertices.at(0), vertices.at(1), vertices.at(2) );
      tri_new = (new Triangle( vertices.at(0), vertices.at(1), vertices.at(2), texture, uv, solid_fraction, currentUUID ));
    }
    float transform[16];
    p->getTransformationMatrix(transform);
    tri_new->setTransformationMatrix(transform);
    primitives[currentUUID] = tri_new;
  }else if( type==PRIMITIVE_TYPE_VOXEL ){
    Voxel* p = getVoxelPointer(UUID);
    Voxel* voxel_new;
    if( !p->hasTexture() ){
      voxel_new = (new Voxel( p->getColorRGBA(), currentUUID ));
    }else{
      voxel_new = (new Voxel( p->getColorRGBA(), currentUUID ));
      /* \todo Texture-mapped voxels constructor here */
    }
    float transform[16];
    p->getTransformationMatrix(transform);
    voxel_new->setTransformationMatrix(transform);
    primitives[currentUUID] = voxel_new;
  }
  
  markGeometryDirty();
  currentUUID++;
  return currentUUID-1;
}

Primitive* Context::getPrimitivePointer( uint UUID ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPrimitivePointer): UUID of " << UUID << " out of range (only " << primitives.size() << " primitives in the context)." << std::endl;
    exit(EXIT_FAILURE);
  }
  return primitives.at(UUID);
}

bool Context::doesPrimitiveExist( const uint UUID ) const{
  return primitives.find(UUID) != primitives.end();
}

Patch* Context::getPatchPointer( const uint UUID ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getPatchPointer): UUID of " << UUID << " not found in the context." << std::endl;
    exit(EXIT_FAILURE);
  }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
    std::cerr << "ERROR (getPatchPointer): Primitive " << UUID << " is not a patch." << std::endl;
  }
  return static_cast<Patch*>(primitives.at(UUID));
}
Triangle* Context::getTrianglePointer( const uint UUID ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getTrianglePointer): UUID of " << UUID << " not found in the context." << std::endl;
    exit(EXIT_FAILURE);
  }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
    std::cerr << "ERROR (getTrianglePointer): Primitive " << UUID << " is not a triangle." << std::endl;
  }
  return static_cast<Triangle*>(primitives.at(UUID));
}

Voxel* Context::getVoxelPointer( const uint UUID ) const{
  if( primitives.find(UUID) == primitives.end() ){
    std::cerr << "ERROR (getVoxelPointer): UUID of " << UUID << " not found in the context." << std::endl;
    exit(EXIT_FAILURE);
  }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
    std::cerr << "ERROR (getVoxelPointer): Primitive " << UUID << " is not a voxel." << std::endl;
  }
  return static_cast<Voxel*>(primitives.at(UUID));
}

uint Context::getPrimitiveCount( void ) const{
  return primitives.size();
}

std::vector<uint> Context::getAllUUIDs( void ) const{
  std::vector<uint> UUIDs;
  UUIDs.resize(primitives.size());
  size_t i=0;
  for(std::map<uint,Primitive*>::const_iterator iter = primitives.begin(); iter != primitives.end(); ++iter){
    UUIDs.at(i) = iter->first;
    i++;
  }
  return UUIDs;
}

void Context::addTimeseriesData( const char* label, float value, Date date, Time time ){

  //floating point value corresponding to date and time
  double date_value = floor(date.year*366.25) + date.JulianDay();
  date_value += double(time.hour)/24. + double(time.minute)/1440. + double(time.second)/86400.;

  //Check if data label already exists
  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    timeseries_data[label].push_back( value );
    timeseries_datevalue[label].push_back( date_value );
    return;
  }else{ //exists

    uint N = getTimeseriesLength(label);

    std::vector<float>::iterator it_data = timeseries_data[label].begin();
    std::vector<double>::iterator it_datevalue = timeseries_datevalue[label].begin();

    if( N==1 ){

      if( date_value<timeseries_datevalue[label].front() ){
	timeseries_data[label].insert(it_data, value);
	timeseries_datevalue[label].insert(it_datevalue, date_value);
	return;
      }else{
	timeseries_data[label].insert(it_data+1, value);
	timeseries_datevalue[label].insert(it_datevalue+1, date_value);
	return;
      }

    }else{

      if( date_value<timeseries_datevalue[label].front() ){ //check if data should be inserted at beginning of timeseries
	timeseries_data[label].insert(it_data, value);
	timeseries_datevalue[label].insert(it_datevalue, date_value);
	return;
      }else if( date_value>timeseries_datevalue[label].back() ){ //check if data should be inserted at end of timeseries
	timeseries_data[label].push_back( value);
	timeseries_datevalue[label].push_back( date_value);
	return;
      }

      //data should be inserted somewhere in the middle of timeseries
      for( uint t=0; t<N-1; t++ ){
	if( date_value==timeseries_datevalue[label].at(t) ){
	  std::cout << "WARNING (addTimeseriesData): Skipping duplicate timeseries date/time." << std::endl;
	  continue;
	}
	if( date_value>timeseries_datevalue[label].at(t) && date_value<timeseries_datevalue[label].at(t+1) ){
	  timeseries_data[label].insert(it_data+t+1, value);
	  timeseries_datevalue[label].insert(it_datevalue+t+1, date_value);
	  return;
	}
      }

    }

  }
  
  std::cerr << "ERROR (addTimeseriesData): Failed to insert timeseries data for unknown reason." << std::endl;
  exit(EXIT_FAILURE);

}

void Context::setCurrentTimeseriesPoint( const char* label, const uint index ){
  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    std::cerr << "ERROR (setCurrentTimeseriesPoint): Timeseries variable `" << label << "' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }
  setDate( queryTimeseriesDate( label, index ) );
  setTime( queryTimeseriesTime( label, index ) );
}

float Context::queryTimeseriesData( const char* label, const Date date, const Time time ) const{
  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    std::cerr << "ERROR (queryTimeseriesData): Timeseries variable `" << label << "' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  double date_value = floor(date.year*366.25) + date.JulianDay();
  date_value += double(time.hour)/24. + double(time.minute)/1440. + double(time.second)/86400.;

  double tmin = timeseries_datevalue.at(label).front();
  double tmax = timeseries_datevalue.at(label).back();

  if( date_value<tmin ){
    std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the earliest data point in the timeseries." << std::endl;
    return timeseries_data.at(label).front();
  }else if( date_value>tmax ){
    std::cout << date_value << " " << tmax << std::endl;
    std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the latest data point in the timeseries." << std::endl;
    return timeseries_data.at(label).back();
  }

  if( timeseries_datevalue.at(label).size() == 0 ){
    std::cout << "WARNING (queryTimeseriesData): timeseries " << label << " does not contain any data." << std::endl;
    return 0;
  }else if( timeseries_datevalue.at(label).size() == 1 ){
    return timeseries_data.at(label).front();
  }else{
    int i;
    bool success=false;
    for( i=0; i<timeseries_data.at(label).size()-1; i++ ){
      if( date_value>=timeseries_datevalue.at(label).at(i) && date_value<=timeseries_datevalue.at(label).at(i+1) ){
	success = true;
	break;
      }
    }

    if(!success){
      
    exit(EXIT_FAILURE);
    }

    double xminus = timeseries_data.at(label).at(i);
    double xplus = timeseries_data.at(label).at(i+1);

    double tminus = timeseries_datevalue.at(label).at(i);
    double tplus = timeseries_datevalue.at(label).at(i+1);

    return xminus + (xplus-xminus)*(date_value-tminus)/(tplus-tminus);
    
  }

}

float Context::queryTimeseriesData( const char* label, const uint index ) const{

  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    std::cerr << "ERROR (getTimeseriesData): Timeseries variable `" << label << "' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }
  
  return timeseries_data.at(label).at(index);
  
}

Time Context::queryTimeseriesTime( const char* label, const uint index ) const{

  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    std::cerr << "ERROR (getTimeseriesTime): Timeseries variable `" << label << "' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  double dateval = timeseries_datevalue.at(label).at(index);
  
  int year = floor(floor(dateval)/366.25);
  assert( year>1000 && year<10000 );

  int JD = floor(dateval-floor(double(year)*366.25));
  assert( JD>0 && JD<367 );

  int hour = floor((dateval-floor(dateval))*24.);
  int minute = floor( ((dateval-floor(dateval))*24.-double(hour))*60. );
  int second = round( ( ( ( dateval - floor(dateval) )*24. - double(hour))*60. - double(minute) )*60.);
  
  if( second==60 ){
    second = 0;
    minute ++;
  }
  
  if( minute==60 ){
    minute = 0;
    hour ++;
  }
  
  assert( second>=0 && second<60 );
  assert( minute>=0 && minute<60 );
  assert( hour>=0 && hour<24 );
    
  return make_Time(hour,minute,second);
  
}

Date Context::queryTimeseriesDate( const char* label, const uint index ) const{

  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    std::cerr << "ERROR (getTimeseriesDate): Timeseries variable `" << label << "' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  double dateval = timeseries_datevalue.at(label).at(index);

  int year = floor(floor(dateval)/366.25);
  assert( year>1000 && year<10000 );

  int JD = floor(dateval-floor(double(year)*366.25));
  assert( JD>0 && JD<367 );

  Date date = Julian2Calendar(JD,year);
  
  return Julian2Calendar(JD,year);
  
}

uint Context::getTimeseriesLength( const char* label ) const{

  if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
    std::cerr << "ERROR (getTimeseriesLength): Timeseries variable `" << label << "' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }else{
    return timeseries_data.at(label).size();
  }

}

void Context::getDomainBoundingBox( vec2& xbounds, vec2& ybounds, vec2& zbounds ) const{

  xbounds.x = 1e8;
  xbounds.y = -1e8;
  ybounds.x = 1e8;
  ybounds.y = -1e8;
  zbounds.x = 1e8;
  zbounds.y = -1e8;

  for( std::map<uint,Primitive*>::const_iterator iter = primitives.begin(); iter != primitives.end(); ++iter ){

    std::vector<vec3> verts = getPrimitivePointer(iter->first)->getVertices();
    
    for( int j=0; j<verts.size(); j++ ){
      if( verts[j].x<xbounds.x ){
	xbounds.x = verts[j].x;
      }else if( verts[j].x>xbounds.y ){
	xbounds.y = verts[j].x;
      }
      if( verts[j].y<ybounds.x ){
	ybounds.x = verts[j].y;
      }else if( verts[j].y>ybounds.y ){
	ybounds.y = verts[j].y;
      }
      if( verts[j].z<zbounds.x ){
	zbounds.x = verts[j].z;
      }else if( verts[j].z>zbounds.y ){
	zbounds.y = verts[j].z;
      }
    }

  }

  return;

}

void Context::getDomainBoundingBox( const std::vector<uint>& UUIDs, vec2& xbounds, vec2& ybounds, vec2& zbounds ) const{

  xbounds.x = 1e8;
  xbounds.y = -1e8;
  ybounds.x = 1e8;
  ybounds.y = -1e8;
  zbounds.x = 1e8;
  zbounds.y = -1e8;

  for( int p=0; p<UUIDs.size(); p++ ){

    std::vector<vec3> verts = getPrimitivePointer( UUIDs.at(p) )->getVertices();
    
    for( int j=0; j<verts.size(); j++ ){
      if( verts[j].x<xbounds.x ){
	xbounds.x = verts[j].x;
      }else if( verts[j].x>xbounds.y ){
	xbounds.y = verts[j].x;
      }
      if( verts[j].y<ybounds.x ){
	ybounds.x = verts[j].y;
      }else if( verts[j].y>ybounds.y ){
	ybounds.y = verts[j].y;
      }
      if( verts[j].z<zbounds.x ){
	zbounds.x = verts[j].z;
      }else if( verts[j].z>zbounds.y ){
	zbounds.y = verts[j].z;
      }
    }

  }

  return;

}

void Context::getDomainBoundingSphere( vec3& center, float& radius ) const{

  vec2 xbounds, ybounds, zbounds;
  getDomainBoundingBox( xbounds, ybounds, zbounds );

  center.x = xbounds.x+0.5f*(xbounds.y - xbounds.x);
  center.y = ybounds.x+0.5f*(ybounds.y - ybounds.x);
  center.z = zbounds.x+0.5f*(zbounds.y - zbounds.x);

  radius = 0.5f*sqrtf( pow(xbounds.y-xbounds.x,2) + pow(ybounds.y-ybounds.x,2) + pow((zbounds.y-zbounds.x),2) );


}

void Context::getDomainBoundingSphere( const std::vector<uint>& UUIDs, vec3& center, float& radius ) const{

  vec2 xbounds, ybounds, zbounds;
  getDomainBoundingBox( UUIDs, xbounds, ybounds, zbounds );

  center.x = xbounds.x+0.5f*(xbounds.y - xbounds.x);
  center.y = ybounds.x+0.5f*(ybounds.y - ybounds.x);
  center.z = zbounds.x+0.5f*(zbounds.y - zbounds.x);

  radius = 0.5f*sqrtf( pow(xbounds.y-xbounds.x,2) + pow(ybounds.y-ybounds.x,2) + pow((zbounds.y-zbounds.x),2) );


}

std::vector<uint> Context::addSphere( const uint Ndivs, const vec3 center, const float radius ){
 
  RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

  return addSphere(Ndivs,center,radius,color);

}

std::vector<uint> Context::addSphere( const uint Ndivs, const vec3 center, const float radius, const RGBcolor color ){
  // return addSphere( Ndivs, center, radius, make_RGBAcolor(color,1.f) );
// }

// std::vector<uint> Context::addSphere( const uint Ndivs, float center, float radius, RGBAcolor color ){

  std::vector<uint> UUID;

  float theta;
  float dtheta=M_PI/float(Ndivs);
  float dphi=2.0*M_PI/float(Ndivs);

  //bottom cap
  for( int j=0; j<Ndivs; j++ ){
      
    vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI, 0 ) );
    vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+dtheta, float(j)*dphi ) );
    vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+dtheta, float(j+1)*dphi ) );
    
    UUID.push_back( addTriangle(v0,v1,v2,color) );

  }

  //top cap
  for( int j=0; j<Ndivs; j++ ){
      
    vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, 0.5*M_PI, 0 ) );
    vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, 0.5*M_PI-dtheta, float(j)*dphi ) );
    vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, 0.5*M_PI-dtheta, float(j+1)*dphi ) );
    
    UUID.push_back( addTriangle(v2,v1,v0,color) );

  }

  //middle
  for( int j=0; j<Ndivs; j++ ){
    for( int i=1; i<Ndivs-1; i++ ){
      
      vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i)*dtheta, float(j)*dphi ) );
      vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i+1)*dtheta, float(j)*dphi ) );
      vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i+1)*dtheta, float(j+1)*dphi ) );
      vec3 v3 = center + sphere2cart( make_SphericalCoord(radius, -0.5*M_PI+float(i)*dtheta, float(j+1)*dphi ) );
	
      UUID.push_back( addTriangle(v0,v1,v2,color) );
      UUID.push_back( addTriangle(v0,v2,v3,color) );      
      
    }
  }

  return UUID;


}

// std::vector<uint> Context::addSphere( const uint Ndivs, std::vector<vec3> nodes, float radius, const char* texture_file ){}

std::vector<uint> Context::addTube( const uint Ndivs, const std::vector<vec3> nodes, const std::vector<float> radius ){

  uint node_count = nodes.size();
  
  std::vector<RGBcolor> color;
  color.resize(node_count);

  for( uint i=0; i<node_count; i++ ){
    color.at(i) = make_RGBcolor(0.f,0.75f,0.f); //Default color is green
  }

  return addTube(Ndivs,nodes,radius,color);

}

std::vector<uint> Context::addTube( const uint Ndivs, const std::vector<vec3> nodes, const std::vector<float> radius, std::vector<RGBcolor> color ){

  const uint node_count = nodes.size();

  if( node_count==0 ){
    std::cerr << "ERROR (Context::addTube): Node and radius arrays are empty." << std::endl;
    exit(EXIT_FAILURE);
  }else if( node_count!=radius.size() ){
    std::cerr << "ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree." << std::endl;
    exit(EXIT_FAILURE);
  }else if( node_count!=color.size() ){
    std::cerr << "ERROR (Context::addTube): Size of `nodes' and `color' arguments must agree." << std::endl;
    exit(EXIT_FAILURE);
  }

  vec3 vec, convec;
  float cfact[Ndivs+1], sfact[Ndivs+1];
  std::vector<std::vector<vec3> > xyz, normal;
  xyz.resize(Ndivs+1);
  normal.resize(Ndivs+1);
  for( uint j=0; j<Ndivs+1; j++ ){
    xyz.at(j).resize(node_count);
    normal.at(j).resize(node_count);
  }
  vec3 nvec(0.1817f,0.6198f,0.7634f);//random vector to get things going

  for( int j=0; j<Ndivs+1; j++ ){
    cfact[j]=cos(2.f*M_PI*float(j)/float(Ndivs));
    sfact[j]=sin(2.f*M_PI*float(j)/float(Ndivs));
  }

  for( int i=0; i<node_count; i++ ){ //looping over tube segments

    if(i==0){
      vec.x=nodes[i+1].x-nodes[i].x;
      vec.y=nodes[i+1].y-nodes[i].y;
      vec.z=nodes[i+1].z-nodes[i].z;
    }else if(i==node_count-1){
      vec.x=nodes[i].x-nodes[i-1].x;
      vec.y=nodes[i].y-nodes[i-1].y;
      vec.z=nodes[i].z-nodes[i-1].z;
    }else{
      vec.x=0.5f*( (nodes[i].x-nodes[i-1].x)+(nodes[i+1].x-nodes[i].x) );
      vec.y=0.5f*( (nodes[i].y-nodes[i-1].y)+(nodes[i+1].y-nodes[i].y) );
      vec.z=0.5f*( (nodes[i].z-nodes[i-1].z)+(nodes[i+1].z-nodes[i].z) );
    }
    
    float norm;
    convec = cross(nvec,vec);
    norm=convec.magnitude();
    convec.x=convec.x/norm;
    convec.y=convec.y/norm;
    convec.z=convec.z/norm;
    nvec = cross(vec,convec);
    norm=nvec.magnitude();
    nvec.x=nvec.x/norm;
    nvec.y=nvec.y/norm;
    nvec.z=nvec.z/norm;

    
    
    for( int j=0; j<Ndivs+1; j++ ){
      normal[j][i].x=cfact[j]*radius[i]*nvec.x+sfact[j]*radius[i]*convec.x;
      normal[j][i].y=cfact[j]*radius[i]*nvec.y+sfact[j]*radius[i]*convec.y;
      normal[j][i].z=cfact[j]*radius[i]*nvec.z+sfact[j]*radius[i]*convec.z;
      
      xyz[j][i].x=nodes[i].x+normal[j][i].x;
      xyz[j][i].y=nodes[i].y+normal[j][i].y;
      xyz[j][i].z=nodes[i].z+normal[j][i].z;
      
      normal[j][i] = normal[j][i]/radius[i];
    }

  }

  vec3 v0, v1, v2;
  std::vector<uint> UUID;

  for( int i=0; i<node_count-1; i++ ){
    for( int j=0; j<Ndivs; j++ ){

      v0 = xyz[j][i];
      v1 = xyz[j+1][i+1];
      v2 = xyz[j+1][i];

      //if( (v1-v0).magnitude()>1e-6 && (v2-v0).magnitude()>1e-6 && (v2-v1).magnitude()>1e-6 ){
	UUID.push_back(addTriangle( v0, v1, v2, color.at(i) ));
	//}
      
      v0 = xyz[j][i];
      v1 = xyz[j][i+1];
      v2 = xyz[j+1][i+1];

      //if( (v1-v0).magnitude()>1e-6 && (v2-v0).magnitude()>1e-6 && (v2-v1).magnitude()>1e-6 ){
	UUID.push_back(addTriangle( v0, v1, v2, color.at(i) ));
	//}
      
    }
  }

  return UUID;

}

std::vector<uint> Context::addTube( const uint Ndivs, const std::vector<vec3> nodes, const std::vector<float> radius, const char* texturefile ){

  const uint node_count = nodes.size();

  if( node_count==0 ){
    std::cerr << "ERROR (Context::addTube): Node and radius arrays are empty." << std::endl;
    exit(EXIT_FAILURE);
  }else if( node_count!=radius.size() ){
    std::cerr << "ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree." << std::endl;
    exit(EXIT_FAILURE);
  }

  vec3 vec, convec;
  float cfact[Ndivs+1], sfact[Ndivs+1];
  std::vector<std::vector<vec3> > xyz, normal;
  std::vector<std::vector<vec2> > uv;
  xyz.resize(Ndivs+1);
  normal.resize(Ndivs+1);
  uv.resize(Ndivs+1);
  for( uint j=0; j<Ndivs+1; j++ ){
    xyz.at(j).resize(node_count);
    normal.at(j).resize(node_count);
    uv.at(j).resize(node_count);
  }
  vec3 nvec(0.f,1.f,0.f);

  for( int j=0; j<Ndivs+1; j++ ){
    cfact[j]=cos(2.f*M_PI*float(j)/float(Ndivs));
    sfact[j]=sin(2.f*M_PI*float(j)/float(Ndivs));
  }

  for( int i=0; i<node_count; i++ ){ //looping over tube segments

    if(i==0){
      vec.x=nodes[i+1].x-nodes[i].x;
      vec.y=nodes[i+1].y-nodes[i].y;
      vec.z=nodes[i+1].z-nodes[i].z;
    }else if(i==node_count-1){
      vec.x=nodes[i].x-nodes[i-1].x;
      vec.y=nodes[i].y-nodes[i-1].y;
      vec.z=nodes[i].z-nodes[i-1].z;
    }else{
      vec.x=0.5f*( (nodes[i].x-nodes[i-1].x)+(nodes[i+1].x-nodes[i].x) );
      vec.y=0.5f*( (nodes[i].y-nodes[i-1].y)+(nodes[i+1].y-nodes[i].y) );
      vec.z=0.5f*( (nodes[i].z-nodes[i-1].z)+(nodes[i+1].z-nodes[i].z) );
    }
    
    float norm;
    convec = cross(nvec,vec);
    norm=convec.magnitude();
    convec.x=convec.x/norm;
    convec.y=convec.y/norm;
    convec.z=convec.z/norm;
    nvec = cross(vec,convec);
    norm=nvec.magnitude();
    nvec.x=nvec.x/norm;
    nvec.y=nvec.y/norm;
    nvec.z=nvec.z/norm;
    
    for( int j=0; j<Ndivs+1; j++ ){
      normal[j][i].x=cfact[j]*radius[i]*nvec.x+sfact[j]*radius[i]*convec.x;
      normal[j][i].y=cfact[j]*radius[i]*nvec.y+sfact[j]*radius[i]*convec.y;
      normal[j][i].z=cfact[j]*radius[i]*nvec.z+sfact[j]*radius[i]*convec.z;
      
      xyz[j][i].x=nodes[i].x+normal[j][i].x;
      xyz[j][i].y=nodes[i].y+normal[j][i].y;
      xyz[j][i].z=nodes[i].z+normal[j][i].z;
      
      uv[j][i].x = float(i)/float(node_count-1);
      uv[j][i].y = float(j)/float(Ndivs);

      normal[j][i] = normal[j][i]/radius[i];
    }

  }

  vec3 v0, v1, v2;
  vec2 uv0, uv1, uv2;
  std::vector<uint> UUID;

  for( int i=0; i<node_count-1; i++ ){
    for( int j=0; j<Ndivs; j++ ){

      v0 = xyz[j][i];
      v1 = xyz[j+1][i+1];
      v2 = xyz[j+1][i];

      uv0 = uv[j][i];
      uv1 = uv[j+1][i+1];
      uv2 = uv[j+1][i];

      if( (v1-v0).magnitude()>1e-6 && (v2-v0).magnitude()>1e-6 && (v2-v1).magnitude()>1e-6 ){
	UUID.push_back(addTriangle( v0, v1, v2, texturefile, uv0, uv1, uv2 ));
      }
      
      v0 = xyz[j][i];
      v1 = xyz[j][i+1];
      v2 = xyz[j+1][i+1];

      uv0 = uv[j][i];
      uv1 = uv[j][i+1];
      uv2 = uv[j+1][i+1];

      if( (v1-v0).magnitude()>1e-6 && (v2-v0).magnitude()>1e-6 && (v2-v1).magnitude()>1e-6 ){
	UUID.push_back(addTriangle( v0, v1, v2, texturefile, uv0, uv1, uv2 ));
      }

    }
  }

  return UUID;

}

std::vector<uint> Context::addBox( const vec3 center, const vec3 size, const int3 subdiv ){

  RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

  return addBox(center,size,subdiv,color,false);
}

std::vector<uint> Context::addBox( const vec3 center, const vec3 size, const int3 subdiv, const RGBcolor color ){
  return addBox(center,size,subdiv,color,false);
}

std::vector<uint> Context::addBox( const vec3 center, const vec3 size, const int3 subdiv, const char* texturefile ){
  return addBox(center,size,subdiv,texturefile,false);
}

std::vector<uint> Context::addBox( const vec3 center, const vec3 size, const int3 subdiv, const RGBcolor color, const bool reverse_normals ){

  std::vector<uint> UUID;

  vec3 subsize;
  subsize.x = size.x/float(subdiv.x);
  subsize.y = size.y/float(subdiv.y);
  subsize.z = size.z/float(subdiv.z);

  vec3 subcenter;
  std::vector<uint> U;

  if( reverse_normals ){ //normals point inward

    // x-z faces (vertical)

    //right
    subcenter = center + make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	
    //left
    subcenter = center - make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );

    // y-z faces (vertical)
	
    //front
    subcenter = center + make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	
    //back
    subcenter = center - make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
    
    // x-y faces (horizontal)

    //top
    subcenter = center + make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );

    //bottom
    subcenter = center - make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );

  }else{ //normals point outward

    // x-z faces (vertical)
 	
    //right
    subcenter = center + make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	
    //left
    subcenter = center - make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	    
    // y-z faces (vertical)
      
    //front
    subcenter = center + make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
    
    //back
    subcenter = center - make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	    
    // x-y faces (horizontal)
	
    //top
    subcenter = center + make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );
    
    //bottom
    subcenter = center - make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), color );
    UUID.insert( UUID.end(), U.begin(), U.end() );

  }
  
  return UUID;

}

std::vector<uint> Context::addBox( const vec3 center, const vec3 size, const int3 subdiv, const char* texturefile, const bool reverse_normals ){

  std::vector<uint> UUID;

  vec3 subsize;
  subsize.x = size.x/float(subdiv.x);
  subsize.y = size.y/float(subdiv.y);
  subsize.z = size.z/float(subdiv.z);

  vec3 subcenter;
  std::vector<uint> U;

  if( reverse_normals ){ //normals point inward

    // x-z faces (vertical)

    //right
    subcenter = center + make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	
    //left
    subcenter = center - make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );

    // y-z faces (vertical)
	
    //front
    subcenter = center + make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	
    //back
    subcenter = center - make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
    
    // x-y faces (horizontal)

    //top
    subcenter = center + make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );

    //bottom
    subcenter = center - make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );

  }else{ //normals point outward

    // x-z faces (vertical)
 	
    //right
    subcenter = center + make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	
    //left
    subcenter = center - make_vec3(0,0.5*size.y,0);
    U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	    
    // y-z faces (vertical)
      
    //front
    subcenter = center + make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
    
    //back
    subcenter = center - make_vec3(0.5*size.x,0,0);
    U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
	    
    // x-y faces (horizontal)
	
    //top
    subcenter = center + make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );
    
    //bottom
    subcenter = center - make_vec3(0,0,0.5*size.z);
    U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), texturefile );
    UUID.insert( UUID.end(), U.begin(), U.end() );

  }
  
  return UUID;

}

std::vector<uint> Context::addTile( const vec3 center, const vec2 size, const SphericalCoord rotation, const int2 subdiv ){

  RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

  return addTile(center,size,rotation,subdiv,color);
}

std::vector<uint> Context::addTile( const vec3 center, const vec2 size, const SphericalCoord rotation, const int2 subdiv, const RGBcolor color ){

  std::vector<uint> UUID;

  vec2 subsize;
  subsize.x = size.x/float(subdiv.x);
  subsize.y = size.y/float(subdiv.y);

  vec3 subcenter;

  for( uint j=0; j<subdiv.y; j++ ){
    for( uint i=0; i<subdiv.x; i++ ){
      
      subcenter = make_vec3(-0.5*size.x+(float(i)+0.5)*subsize.x,-0.5*size.y+(float(j)+0.5)*subsize.y,0);
            
      UUID.push_back( addPatch( subcenter, subsize, make_SphericalCoord(0,0), color ) );

      if( rotation.elevation!=0.f ){
	getPrimitivePointer( UUID.back() )->rotate( -rotation.elevation, "x" );
      }
      if( rotation.azimuth!=0.f ){
	getPrimitivePointer( UUID.back() )->rotate( -rotation.azimuth, "z" );
      }
      getPrimitivePointer( UUID.back() )->translate( center );

    }
  }

  return UUID;

}

std::vector<uint> Context::addTile( const vec3 center, const vec2 size, const SphericalCoord rotation, const int2 subdiv, const char* texturefile ){

  std::vector<uint> UUID;

  vec2 subsize; 
  subsize.x = size.x/float(subdiv.x);
  subsize.y = size.y/float(subdiv.y);

  vec3 subcenter;

  std::vector<helios::vec2> uv;
  uv.resize(4);
  vec2 uv_sub;
  uv_sub.x = 1.f/float(subdiv.x);
  uv_sub.y = 1.f/float(subdiv.y);

  Texture* texture = addTexture( texturefile );
  std::vector<std::vector<bool> >* alpha;
  int2 sz;
  if( texture->hasTransparencyChannel() ){
    alpha = texture->getTransparencyData();
    sz = texture->getSize();
  }

  for( uint j=0; j<subdiv.y; j++ ){
    for( uint i=0; i<subdiv.x; i++ ){
      
      subcenter = make_vec3(-0.5*size.x+(float(i)+0.5)*subsize.x,-0.5*size.y+(float(j)+0.5)*subsize.y,0);

      uv.at(0) = make_vec2(1.f-float(i)*uv_sub.x,float(j)*uv_sub.y);
      uv.at(1) = make_vec2(1.f-float(i+1)*uv_sub.x,float(j)*uv_sub.y); 
      uv.at(2) = make_vec2(1.f-float(i+1)*uv_sub.x,float(j+1)*uv_sub.y);
      uv.at(3) = make_vec2(1.f-float(i)*uv_sub.x,float(j+1)*uv_sub.y);

      float solid_fraction;
      if( texture->hasTransparencyChannel() ){
	int A = 0;
	int At = 0;
	
	int2 uv_min( floor(uv.at(1).x*(sz.x-1)), floor(uv.at(1).y*(sz.y-1)) );
	int2 uv_max( floor(uv.at(3).x*(sz.x-1)), floor(uv.at(3).y*(sz.y-1)) );

	assert( uv_min.x>=0 && uv_min.y>=0 && uv_max.x<sz.x && uv_max.y<sz.y );
	
	for( int j=uv_min.y; j<uv_max.y; j++ ){
	  for( int i=uv_min.x; i<uv_max.x; i++ ){
	    At += 1;
	    if( alpha->at(j).at(i) ){
	      A += 1;
	    }
	  }
	}
	if( At==0 ){
	  solid_fraction = 0;
	}else{
	  solid_fraction = float(A)/float(At);
	}
      }else{
	solid_fraction = 1.f;
      }

      Patch* patch_new = (new Patch( texture, uv, solid_fraction, currentUUID ));

      assert( size.x>0.f && size.y>0.f );
      patch_new->scale( make_vec3(subsize.x,subsize.y,1) );

      patch_new->translate( subcenter );
      
      if( rotation.elevation!=0 ){
	patch_new->rotate(-rotation.elevation, "x");
      }
      if( rotation.azimuth!=0 ){
	patch_new->rotate(-rotation.azimuth, "z");
      }
      
      patch_new->translate( center );
  
      primitives[currentUUID] = patch_new;
      markGeometryDirty();
      currentUUID++;
      UUID.push_back(currentUUID-1);


    }
  }

  return UUID;

}

std::vector<uint> Context::addDisk( const uint Ndivs, const vec3& center, const vec2& size ){
  return addDisk(Ndivs,center,size,make_SphericalCoord(0,0),make_RGBAcolor(1,0,0,1));
}

std::vector<uint> Context::addDisk( const uint Ndivs, const vec3& center, const vec2& size, const SphericalCoord& rotation ){
  return addDisk(Ndivs,center,size,rotation,make_RGBAcolor(1,0,0,1));
}

std::vector<uint> Context::addDisk( const uint Ndivs, const vec3& center, const vec2& size, const SphericalCoord& rotation, const RGBcolor& color ){
  return addDisk(Ndivs,center,size,rotation,make_RGBAcolor(color,1));
}

std::vector<uint> Context::addDisk( const uint Ndivs, const vec3& center, const vec2& size, const SphericalCoord& rotation, const RGBAcolor& color ){

  std::vector<uint> UUIDs;
  UUIDs.resize(Ndivs);
  
  for( int i=0; i<Ndivs; i++ ){

    float dtheta = 1.f/float(Ndivs)*2.f*M_PI;
    
    UUIDs.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), color );
    getPrimitivePointer(UUIDs.at(i))->rotate( rotation.elevation, "y" );
    getPrimitivePointer(UUIDs.at(i))->rotate( rotation.azimuth, "z" );
    getPrimitivePointer(UUIDs.at(i))->translate( center );
    
  }

  return UUIDs;

}

std::vector<uint> Context::addDisk( const uint Ndivs, const vec3& center, const vec2& size, const SphericalCoord& rotation, const char* texture_file ){

  std::vector<uint> UUIDs;
  UUIDs.resize(Ndivs);
  
  for( int i=0; i<Ndivs; i++ ){

    float dtheta = 1.f/float(Ndivs)*2.f*M_PI;
    
    UUIDs.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), texture_file, make_vec2(0.5,0.5), make_vec2(0.5*(1+cosf(dtheta*float(i))),0.5*(1+sinf(dtheta*float(i)))), make_vec2(0.5*(1+cosf(dtheta*float(i+1))),0.5*(1+sinf(dtheta*float(i+1))))  );
    getPrimitivePointer(UUIDs.at(i))->rotate( rotation.elevation, "y" );
    getPrimitivePointer(UUIDs.at(i))->rotate( rotation.azimuth, "z" );
    getPrimitivePointer(UUIDs.at(i))->translate( center );
    
  }

  return UUIDs;

} 

void Context::loadPData( pugi::xml_node p, uint UUID ){

  for (pugi::xml_node data = p.child("data_int"); data; data = data.next_sibling("data_int")){

      const char* data_str = data.child_value();
      std::vector<int> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	int tmp;
	while( data_stream >> tmp ){
	  datav.push_back(tmp);
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_INT,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_uint"); data; data = data.next_sibling("data_uint")){

      const char* data_str = data.child_value();
      std::vector<uint> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	uint tmp;
	while( data_stream >> tmp ){
	  datav.push_back(tmp);
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_UINT,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_float"); data; data = data.next_sibling("data_float")){

      const char* data_str = data.child_value();
      std::vector<float> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	float tmp;
	while( data_stream >> tmp ){
	  datav.push_back(tmp);
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_FLOAT,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_double"); data; data = data.next_sibling("data_double")){

      const char* data_str = data.child_value();
      std::vector<double> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	double tmp;
	while( data_stream >> tmp ){
	  datav.push_back(tmp);
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_DOUBLE,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_vec2"); data; data = data.next_sibling("data_vec2")){

      const char* data_str = data.child_value();
      std::vector<vec2> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::vector<float> tmp;
	tmp.resize(2);
	int c = 0;
	while( data_stream >> tmp.at(c) ){
	  c++;
	  if( c==2 ){
	    datav.push_back(make_vec2(tmp.at(0),tmp.at(1)));
	    c=0;
	  }	  
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_VEC2,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_vec3"); data; data = data.next_sibling("data_vec3")){

      const char* data_str = data.child_value();
      std::vector<vec3> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::vector<float> tmp;
	tmp.resize(3);
	int c = 0;
	while( data_stream >> tmp.at(c) ){
	  c++;
	  if( c==3 ){
	    datav.push_back(make_vec3(tmp.at(0),tmp.at(1),tmp.at(2)));
	    c=0;
	  }	  
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_VEC3,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_vec4"); data; data = data.next_sibling("data_vec4")){

      const char* data_str = data.child_value();
      std::vector<vec4> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::vector<float> tmp;
	tmp.resize(4);
	int c = 0;
	while( data_stream >> tmp.at(c) ){
	  c++;
	  if( c==4 ){
	    datav.push_back(make_vec4(tmp.at(0),tmp.at(1),tmp.at(2),tmp.at(3)));
	    c=0;
	  }	  
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_VEC4,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_int2"); data; data = data.next_sibling("data_int2")){

      const char* data_str = data.child_value();
      std::vector<int2> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::vector<int> tmp;
	tmp.resize(2);
	int c = 0;
	while( data_stream >> tmp.at(c) ){
	  c++;
	  if( c==2 ){
	    datav.push_back(make_int2(tmp.at(0),tmp.at(1)));
	    c=0;
	  }	  
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_INT2,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_int3"); data; data = data.next_sibling("data_int3")){

      const char* data_str = data.child_value();
      std::vector<int3> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::vector<int> tmp;
	tmp.resize(3);
	int c = 0;
	while( data_stream >> tmp.at(c) ){
	  c++;
	  if( c==3 ){
	    datav.push_back(make_int3(tmp.at(0),tmp.at(1),tmp.at(2)));
	    c=0;
	  }	  
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_INT3,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_int4"); data; data = data.next_sibling("data_int4")){

      const char* data_str = data.child_value();
      std::vector<int4> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::vector<int> tmp;
	tmp.resize(4);
	int c = 0;
	while( data_stream >> tmp.at(c) ){
	  c++;
	  if( c==4 ){
	    datav.push_back(make_int4(tmp.at(0),tmp.at(1),tmp.at(2),tmp.at(3)));
	    c=0;
	  }	  
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_INT4,datav.size(),&datav[0]);
      }
      
    }

    for (pugi::xml_node data = p.child("data_string"); data; data = data.next_sibling("data_string")){

      const char* data_str = data.child_value();
      std::vector<std::string> datav;
      if( strlen(data_str)>0 ){
	std::istringstream data_stream(data_str);
	std::string tmp;
	while( data_stream >> tmp ){
	  datav.push_back(tmp);
	}
      }

      const char* label = data.attribute("label").value();

      if( datav.size()==1 ){
	setPrimitiveData(UUID,label,datav.front());
      }else if( datav.size()>1 ){
	setPrimitiveData(UUID,label,HELIOS_TYPE_STRING,datav.size(),&datav[0]);
      }
      
    }

}

std::vector<uint> Context::loadXML( const char* filename ){

  std::cout << "Loading XML file: " << filename << "..." << std::flush;

  XMLfiles.push_back( filename );

  uint ID;
  std::vector<uint> UUID;

  // Using "pugixml" parser.  See pugixml.org
  pugi::xml_document xmldoc;

  //load file
  pugi::xml_parse_result result = xmldoc.load_file(filename);

  //error checking
  if (!result){
    std::cout << "failed." << std::endl;
    std::cerr << "XML [" << filename << "] parsed with errors, attr value: [" << xmldoc.child("node").attribute("attr").value() << "]\n";
    std::cerr << "Error description: " << result.description() << "\n";
    std::cerr << "Error offset: " << result.offset << " (error at [..." << (filename + result.offset) << "]\n\n";
    exit(EXIT_FAILURE);
  }

  pugi::xml_node helios = xmldoc.child("helios");

  if( helios.empty() ){
    std::cout << "failed." << std::endl;
    std::cerr << "ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags." << std::endl;
    exit(EXIT_FAILURE);
  }

  //-------------- TIME/DATE ---------------//

  for (pugi::xml_node p = helios.child("date"); p; p = p.next_sibling("date")){

    pugi::xml_node year_node = p.child("year");
    const char* year_str = year_node.child_value();
    int year = atoi( year_str );

    pugi::xml_node month_node = p.child("month");
    const char* month_str = month_node.child_value();
    int month = atoi( month_str );

    pugi::xml_node day_node = p.child("day");
    const char* day_str = day_node.child_value();
    int day = atoi( day_str );

    setDate( day, month, year );

  }

  for (pugi::xml_node p = helios.child("time"); p; p = p.next_sibling("time")){
  
    pugi::xml_node hour_node = p.child("hour");
    const char* hour_str = hour_node.child_value();
    int hour = atoi( hour_str );

    pugi::xml_node minute_node = p.child("minute");
    const char* minute_str = minute_node.child_value();
    int minute = atoi( minute_str );

    pugi::xml_node second_node = p.child("second");
    const char* second_str = second_node.child_value();
    int second = atoi( second_str );

    setTime( second, minute, hour );
    
  }

  //-------------- PATCHES ---------------//
  for (pugi::xml_node p = helios.child("patch"); p; p = p.next_sibling("patch")){

    // * Patch Transformation Matrix * //
    float transform[16];
    pugi::xml_node transform_node = p.child("transform");
    
    //note: pugi loads xml data as a character.  need to separate it into 3 floats
    const char* transform_str = transform_node.child_value();
    if( strlen(transform_str)==0 ){
      transform[0]=1.f;transform[1]=0.f;transform[2]=0.f;transform[3]=0.f;
      transform[4]=0.f;transform[5]=1.f;transform[6]=0.f;transform[7]=0.f;
      transform[8]=0.f;transform[9]=0.f;transform[10]=1.f;transform[11]=0.f;
      transform[12]=0.f;transform[13]=0.f;transform[14]=0.f;transform[15]=1.f;
    }else{
       std::istringstream stream(transform_str);
       float tmp;
       int i=0;
       while( stream >> tmp ){
	 transform[i] = tmp;
	 i++;
       }
       if( i!=16 ){
	 std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
	 transform[0]=1.f;transform[1]=0.f;transform[2]=0.f;transform[3]=0.f;
	 transform[4]=0.f;transform[5]=1.f;transform[6]=0.f;transform[7]=0.f;
	 transform[8]=0.f;transform[9]=0.f;transform[10]=1.f;transform[11]=0.f;
	 transform[12]=0.f;transform[13]=0.f;transform[14]=0.f;transform[15]=1.f;
       }
    }

    // * Patch Texture * //
    std::string texture_file;
    pugi::xml_node texture_node = p.child("texture");
    const char* texfile = deblank(texture_node.child_value());
    if( strlen(texfile)==0 ){
      texture_file = "none";
    }else{
      texture_file = texfile;
    }

    // * Patch Texture (u,v) Coordinates * //
    std::vector<vec2> uv;
    pugi::xml_node uv_node = p.child("textureUV");
    const char* texUV = uv_node.child_value();
    if( strlen(texUV)>0 ){
      std::istringstream uv_stream(texUV);
      std::vector<float> tmp;
      tmp.resize(2);
      int c = 0;
      while( uv_stream >> tmp.at(c) ){
	c++;
	if( c==2 ){
	  uv.push_back(make_vec2(tmp.at(0),tmp.at(1)));
	  c=0;
	}	  
      }
      if( c!=0 ){
	std::cerr << "WARNING (loadXML): textureUV for patch does not contain an even number of elements. Skipping..." << std::endl;
	uv.resize(0);
      }
      if( uv.size()!=4 ){
	std::cerr << "WARNING (loadXML): textureUV for patch does not contain four pairs of (u,v) coordinates. Skipping..." << std::endl;
	uv.resize(0);
      }
    }
    
    // * Patch Diffuse Colors * //
    RGBAcolor color;
    pugi::xml_node color_node = p.child("color");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* color_str = color_node.child_value();
    if( strlen(color_str)==0 ){
      color = make_RGBAcolor(0,0,0,1);//assume default color of black
    }else{
      color=string2RGBcolor(color_str);
    }

    // * Add the Patch * //
    if( strcmp(texture_file.c_str(),"none")==0 ){
      ID=addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0,0), color );
    }else{
      if( uv.size()==0 ){
	ID=addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0,0), texture_file.c_str() );
      }else{
	ID=addPatch( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0,0), texture_file.c_str(), 0.5*(uv.at(2)+uv.at(0)), uv.at(2)-uv.at(0) );
      }
    }
    getPrimitivePointer(ID)->setTransformationMatrix(transform);

    UUID.push_back(ID);

    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//
    //This is for backward compatability (<v0.5.3)//
    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//

    // * Patch Centers * //
    vec3 center;
    pugi::xml_node center_node = p.child("center");
    
    //note: pugi loads xml data as a character.  need to separate it into 3 floats
    const char* center_str = center_node.child_value();
    if( strlen(center_str)!=0 ){
       center=string2vec3(center_str);
       getPatchPointer(ID)->translate(center);
    }

    // * Patch Sizes * //
    vec2 size;
    pugi::xml_node size_node = p.child("size");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* size_str = size_node.child_value();
    if( strlen(size_str)!=0 ){
      size=string2vec2(size_str);
      getPatchPointer(ID)->scale(make_vec3(size.x,size.y,1));
    }

    // * Patch Rotations * //
    SphericalCoord rotation;
    pugi::xml_node rotation_node = p.child("rotation");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* rotation_str = rotation_node.child_value();
    if( strlen(rotation_str)!=0 ){
      vec2 rot = string2vec2(rotation_str);
      getPatchPointer(ID)->rotate(rot.x,"y");
      getPatchPointer(ID)->rotate(rot.y,"z");
    }

    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^//

    // * Primitive Data * //

    loadPData( p, ID );

  }//end patches

  //-------------- TRIANGLES ---------------//

  //looping over any triangles specified in XML file
  for (pugi::xml_node tri = helios.child("triangle"); tri; tri = tri.next_sibling("triangle")){

    // * Triangle Transformation Matrix * //
    float transform[16];
    pugi::xml_node transform_node = tri.child("transform");
    
    //note: pugi loads xml data as a character.  need to separate it into 3 floats
    const char* transform_str = transform_node.child_value();
    if( strlen(transform_str)==0 ){
      transform[0]=1.f;transform[1]=0.f;transform[2]=0.f;transform[3]=0.f;
      transform[4]=0.f;transform[5]=1.f;transform[6]=0.f;transform[7]=0.f;
      transform[8]=0.f;transform[9]=0.f;transform[10]=1.f;transform[11]=0.f;
      transform[12]=0.f;transform[13]=0.f;transform[14]=0.f;transform[15]=1.f;
    }else{
      std::istringstream stream(transform_str);
       float tmp;
       int i=0;
       while( stream >> tmp ){
	 transform[i] = tmp;
	 i++;
       }
       if( i!=16 ){
	 std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
	 transform[0]=1.f;transform[1]=0.f;transform[2]=0.f;transform[3]=0.f;
	 transform[4]=0.f;transform[5]=1.f;transform[6]=0.f;transform[7]=0.f;
	 transform[8]=0.f;transform[9]=0.f;transform[10]=1.f;transform[11]=0.f;
	 transform[12]=0.f;transform[13]=0.f;transform[14]=0.f;transform[15]=1.f;
       }
    }

    // * Triangle Texture * //
    std::string texture_file;
    pugi::xml_node texture_node = tri.child("texture");
    const char* texfile = deblank(texture_node.child_value());
    if( strlen(texfile)==0 ){
      texture_file = "none";
    }else{
      texture_file = texfile;
    }

    // * Triangle Texture (u,v) Coordinates * //
    std::vector<vec2> uv;
    pugi::xml_node uv_node = tri.child("textureUV");
    const char* texUV = uv_node.child_value();
    if( strlen(texUV)>0 ){
      std::istringstream uv_stream(texUV);
      std::vector<float> tmp;
      tmp.resize(2);
      int c = 0;
      while( uv_stream >> tmp.at(c) ){
	c++;
	if( c==2 ){
	  uv.push_back(make_vec2(tmp.at(0),tmp.at(1)));
	  c=0;
	}	  
      }
      if( c!=0 ){
	std::cerr << "WARNING (loadXML): textureUV for patch does not contain an even number of elements. Skipping..." << std::endl;
	uv.resize(0);
      }
      if( uv.size()!=3 ){
	std::cerr << "WARNING (loadXML): textureUV for triangle does not contain three pairs of (u,v) coordinates. Skipping..." << std::endl;
	uv.resize(0);
      }
    }

    // * Triangle Diffuse Colors * //
    RGBAcolor color;
    pugi::xml_node color_node = tri.child("color");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* color_str = color_node.child_value();
    if( strlen(color_str)==0 ){
      color = make_RGBAcolor(0,0,0,1);//assume default color of black
    }else{
      color=string2RGBcolor(color_str);
    }

    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//
    //This is for backward compatability (<v0.5.3)//
    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//

    // * Triangle Vertices //
    std::vector<vec3> vert_pos;
    vert_pos.resize(3);

    pugi::xml_node vertices = tri.child("vertex");
    
    for(int i=0;i<3;i++){
    
      //note: pugi loads xml data as a character.  need to separate it into 3 floats
      const char* str = vertices.child_value();
      vert_pos.at(i)=string2vec3(str);

      if( strlen(str)==0 ){
	if( i==0 ){
	  break;
	}else if( i==1 ){
	  std::cout << "failed." << std::endl;
	  std::cerr << "ERROR (loadXML): Only 1 vertex was given for triangle #" << tri << " (requires 3)." << std::endl;
	}else if( i==2) {
	  std::cout << "failed." << std::endl;
	  std::cerr << "ERROR (loadXML): Only 2 vertices were given for triangle #" << tri << " (requires 3)." << std::endl;
	}
	exit(EXIT_FAILURE);
      }
      
      vertices = vertices.next_sibling("vertex");

    }
    
    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^//

    // * Add the Triangle * //
    if( strcmp(texture_file.c_str(),"none")==0 || uv.size()==0 ){
      ID = addTriangle( vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), color );
    }else{
      ID = addTriangle( vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), texture_file.c_str(), uv.at(0), uv.at(1), uv.at(2) );
    }
    getPrimitivePointer(ID)->setTransformationMatrix(transform);

    UUID.push_back(ID);

    // * Primitive Data * //

    loadPData( tri, ID );

  }

  //-------------- VOXELS ---------------//
  for (pugi::xml_node p = helios.child("voxel"); p; p = p.next_sibling("voxel")){

    // * Voxel Transformation Matrix * //
    float transform[16];
    pugi::xml_node transform_node = p.child("transform");
    
    //note: pugi loads xml data as a character.  need to separate it into 3 floats
    const char* transform_str = transform_node.child_value();
    if( strlen(transform_str)==0 ){
      transform[0]=1.f;transform[1]=0.f;transform[2]=0.f;transform[3]=0.f;
      transform[4]=0.f;transform[5]=1.f;transform[6]=0.f;transform[7]=0.f;
      transform[8]=0.f;transform[9]=0.f;transform[10]=1.f;transform[11]=0.f;
      transform[12]=0.f;transform[13]=0.f;transform[14]=0.f;transform[15]=1.f;
    }else{
      std::istringstream stream(transform_str);
       float tmp;
       int i=0;
       while( stream >> tmp ){
	 transform[i] = tmp;
	 i++;
       }
       if( i!=16 ){
	 std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
	 transform[0]=1.f;transform[1]=0.f;transform[2]=0.f;transform[3]=0.f;
	 transform[4]=0.f;transform[5]=1.f;transform[6]=0.f;transform[7]=0.f;
	 transform[8]=0.f;transform[9]=0.f;transform[10]=1.f;transform[11]=0.f;
	 transform[12]=0.f;transform[13]=0.f;transform[14]=0.f;transform[15]=1.f;
       }
    }

    // * Voxel Diffuse Colors * //
    RGBAcolor color;
    pugi::xml_node color_node = p.child("color");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* color_str = color_node.child_value();
    if( strlen(color_str)==0 ){
      color = make_RGBAcolor(0,0,0,1);//assume default color of black
    }else{
      color=string2RGBcolor(color_str);
    }
    
    // * Add the Voxel * //
    ID = addVoxel( make_vec3(0,0,0), make_vec3(0,0,0), 0, color );
    getPrimitivePointer(ID)->setTransformationMatrix(transform);
          
    UUID.push_back(ID);

    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//
    //This is for backward compatability (<v0.5.3)//
    //vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv//

    // * Voxel Centers * //
    vec3 center;
    pugi::xml_node center_node = p.child("center");
    
    //note: pugi loads xml data as a character.  need to separate it into 3 floats
    const char* center_str = center_node.child_value();
    if( strlen(center_str)!=0 ){
      center=string2vec3(center_str);
      getVoxelPointer(ID)->translate(center);
    }

    // * Voxel Sizes * //
    vec3 size;
    pugi::xml_node size_node = p.child("size");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* size_str = size_node.child_value();
    if( strlen(size_str)!=0 ){
      size=string2vec3(size_str);
      getVoxelPointer(ID)->translate(size);
    }

    // * Voxel Rotation * //
    float rotation;
    pugi::xml_node rotation_node = p.child("rotation");
    
    //note: pugi loads xml data as a character.  need to separate it into 2 floats
    const char* rotation_str = rotation_node.child_value();
    if( strlen(rotation_str)!=0 ){
      rotation = atof(rotation_str);
      getVoxelPointer(ID)->rotate(rotation,"z");
    }

    //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^//

    // * Primitive Data * //

    loadPData( p, ID );

  }

  //-------------- GLOBAL DATA ---------------//

  for (pugi::xml_node data = helios.child("globaldata_int"); data; data = data.next_sibling("globaldata_int")){

    const char* data_str = data.child_value();
    std::vector<int> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      int tmp;
      while( data_stream >> tmp ){
	datav.push_back(tmp);
      }
    }

    const char* label = data.attribute("label").value();

    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_INT,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_uint"); data; data = data.next_sibling("globaldata_uint")){

    const char* data_str = data.child_value();
    std::vector<uint> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      uint tmp;
      while( data_stream >> tmp ){
	datav.push_back(tmp);
      }
    }

    const char* label = data.attribute("label").value();

    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_UINT,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_float"); data; data = data.next_sibling("globaldata_float")){

    const char* data_str = data.child_value();
    std::vector<float> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      float tmp;
      while( data_stream >> tmp ){
	datav.push_back(tmp);
      }
    }

    const char* label = data.attribute("label").value();

    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_FLOAT,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_double"); data; data = data.next_sibling("globaldata_double")){

    const char* data_str = data.child_value();
    std::vector<double> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      double tmp;
      while( data_stream >> tmp ){
	datav.push_back(tmp);
      }
    }
    
    const char* label = data.attribute("label").value();
    
    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_DOUBLE,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_vec2"); data; data = data.next_sibling("globaldata_vec2")){

    const char* data_str = data.child_value();
    std::vector<vec2> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::vector<float> tmp;
      tmp.resize(2);
      int c = 0;
      while( data_stream >> tmp.at(c) ){
	c++;
	if( c==2 ){
	  datav.push_back(make_vec2(tmp.at(0),tmp.at(1)));
	  c=0;
	}	  
      }
    }
    
    const char* label = data.attribute("label").value();

    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_VEC2,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_vec3"); data; data = data.next_sibling("globaldata_vec3")){

    const char* data_str = data.child_value();
    std::vector<vec3> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::vector<float> tmp;
      tmp.resize(3);
      int c = 0;
      while( data_stream >> tmp.at(c) ){
	c++;
	if( c==3 ){
	  datav.push_back(make_vec3(tmp.at(0),tmp.at(1),tmp.at(2)));
	  c=0;
	}	  
      }
    }
    
    const char* label = data.attribute("label").value();

    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_VEC3,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_vec4"); data; data = data.next_sibling("globaldata_vec4")){

    const char* data_str = data.child_value();
    std::vector<vec4> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::vector<float> tmp;
      tmp.resize(4);
      int c = 0;
      while( data_stream >> tmp.at(c) ){
	c++;
	if( c==4 ){
	  datav.push_back(make_vec4(tmp.at(0),tmp.at(1),tmp.at(2),tmp.at(3)));
	  c=0;
	}	  
      }
    }
    
    const char* label = data.attribute("label").value();
    
    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_VEC4,datav.size(),&datav[0]);
    }
    
  }
  
  for (pugi::xml_node data = helios.child("globaldata_int2"); data; data = data.next_sibling("globaldata_int2")){
    
    const char* data_str = data.child_value();
    std::vector<int2> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::vector<int> tmp;
      tmp.resize(2);
      int c = 0;
      while( data_stream >> tmp.at(c) ){
	c++;
	if( c==2 ){
	  datav.push_back(make_int2(tmp.at(0),tmp.at(1)));
	  c=0;
	}	  
      }
    }
    
    const char* label = data.attribute("label").value();
    
    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_INT2,datav.size(),&datav[0]);
    }
    
  }
  
  for (pugi::xml_node data = helios.child("globaldata_int3"); data; data = data.next_sibling("globaldata_int3")){

    const char* data_str = data.child_value();
    std::vector<int3> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::vector<int> tmp;
      tmp.resize(3);
      int c = 0;
      while( data_stream >> tmp.at(c) ){
	c++;
	if( c==3 ){
	  datav.push_back(make_int3(tmp.at(0),tmp.at(1),tmp.at(2)));
	  c=0;
	}	  
      }
    }

    const char* label = data.attribute("label").value();

    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_INT3,datav.size(),&datav[0]);
    }
    
  }
  
  for (pugi::xml_node data = helios.child("globaldata_int4"); data; data = data.next_sibling("globaldata_int4")){

    const char* data_str = data.child_value();
    std::vector<int4> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::vector<int> tmp;
      tmp.resize(4);
      int c = 0;
      while( data_stream >> tmp.at(c) ){
	c++;
	if( c==4 ){
	  datav.push_back(make_int4(tmp.at(0),tmp.at(1),tmp.at(2),tmp.at(3)));
	  c=0;
	}	  
      }
    }
    
    const char* label = data.attribute("label").value();
    
    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_INT4,datav.size(),&datav[0]);
    }
    
  }

  for (pugi::xml_node data = helios.child("globaldata_string"); data; data = data.next_sibling("globaldata_string")){

    const char* data_str = data.child_value();
    std::vector<std::string> datav;
    if( strlen(data_str)>0 ){
      std::istringstream data_stream(data_str);
      std::string tmp;
      while( data_stream >> tmp ){
	datav.push_back(tmp);
      }
    }
    
    const char* label = data.attribute("label").value();
    
    if( datav.size()==1 ){
      setGlobalData(label,datav.front());
    }else if( datav.size()>1 ){
      setGlobalData(label,HELIOS_TYPE_STRING,datav.size(),&datav[0]);
    }
    
  }

  //-------------- TIMESERIES DATA ---------------//
  for (pugi::xml_node p = helios.child("timeseries"); p; p = p.next_sibling("timeseries")){

    const char* label = p.attribute("label").value();

    for (pugi::xml_node d = p.child("datapoint"); d; d = d.next_sibling("datapoint")){

      Time time;
      pugi::xml_node time_node = d.child("time");
      const char* time_str = time_node.child_value();
      if( strlen(time_str)>0 ){
	int3 time_ = string2int3(time_str);
	if( time_.x<0 || time_.x>23 ){
	  std::cerr << "ERROR (loadXML): Invalid hour of " << time_.x << " given in timeseries. Hour must be positive and not greater than 23." << std::endl;
	  exit(EXIT_FAILURE);
	}else if( time_.y<0 || time_.y>59 ){
	  std::cerr << "ERROR (loadXML): Invalid minute of " << time_.y << " given in timeseries. Minute must be positive and not greater than 59." << std::endl;
	  exit(EXIT_FAILURE);
	}else if( time_.z<0 || time_.z>59 ){
	  std::cerr << "ERROR (loadXML): Invalid second of " << time_.z << " given in timeseries. Second must be positive and not greater than 59." << std::endl;
	  exit(EXIT_FAILURE);
	}
	time = make_Time(time_.x, time_.y,time_.z);
      }else{
	std::cerr << "ERROR (loadXML): No time was specified for timeseries datapoint." << std::endl;
	exit(EXIT_FAILURE);
      }

      Date date;
      bool date_flag=false;
      
      pugi::xml_node date_node = d.child("date");
      const char* date_str = date_node.child_value();
      if( strlen(date_str)>0 ){
	int3 date_ = string2int3(date_str);
	if( date_.x<1 || date_.x>31 ){
	  std::cerr << "ERROR (loadXML): Invalid day of month " << date_.x << " given in timeseries. Day must be greater than zero and not greater than 31." << std::endl;
	  exit(EXIT_FAILURE);
	}else if( date_.y<1 || date_.y>12 ){
	  std::cerr << "ERROR (loadXML): Invalid month of " << date_.y << " given in timeseries. Month must be greater than zero and not greater than 12." << std::endl;
	  exit(EXIT_FAILURE);
	}else if( date_.z<1000 || date_.z>10000 ){
	  std::cerr << "ERROR (loadXML): Invalid year of " << date_.z << " given in timeseries. Year should be in YYYY format." << std::endl;
	  exit(EXIT_FAILURE);
	}
	date = make_Date(date_.x, date_.y, date_.z );
	date_flag=true;
      }

      pugi::xml_node Jdate_node = d.child("dateJulian");
      const char* Jdate_str = Jdate_node.child_value();
      if( strlen(Jdate_str)>0 ){
	int2 date_ = string2int2(Jdate_str);
	if( date_.x<1 || date_.x>366 ){
	  std::cerr << "ERROR (loadXML): Invalid Julian day of year " << date_.x << " given in timeseries. Julian day must be greater than zero and not greater than 366." << std::endl;
	  exit(EXIT_FAILURE);
	}else if( date_.y<1000 || date_.y>10000 ){
	  std::cerr << "ERROR (loadXML): Invalid year of " << date_.y << " given in timeseries. Year should be in YYYY format." << std::endl;
	  exit(EXIT_FAILURE);
	}
	date = Julian2Calendar( date_.x, date_.y );
	date_flag=true;
      }

      if( !date_flag ){
	std::cerr << "ERROR (loadXML): No date was specified for timeseries datapoint." << std::endl;
	exit(EXIT_FAILURE);
      }

      float value;
      pugi::xml_node value_node = d.child("value");
      const char* value_str = value_node.child_value();
      if( strlen(value_str)>0 ){
	value = atof(value_str);
      }else{
	std::cerr << "ERROR (loadXML): No value was specified for timeseries datapoint." << std::endl;
	exit(EXIT_FAILURE);
      }

      addTimeseriesData(label,value,date,time);

    }
    
  }

  std::cout << "done." << std::endl;

  return UUID;
  
}

std::vector<std::string> Context::getLoadedXMLFiles( void ){
  return XMLfiles;
}

void Context::writeXML( const char* filename ) const{
  std::cout << "Writing XML file " << filename << "..." << std::flush;

  std::ofstream outfile;
  outfile.open(filename);

  outfile << "<?xml version=\"1.0\"?>\n\n";

  outfile << "<helios>\n\n";

  // -- time/date -- //

  Date date = getDate();
  
  outfile << "   <date>" << std::endl;

  outfile << "\t<day>" << date.day << "</day>" << std::endl;
  outfile << "\t<month>" << date.month << "</month>" << std::endl;
  outfile << "\t<year>" << date.year << "</year>" << std::endl;

  outfile << "   </date>" << std::endl;

  Time time = getTime();
  
  outfile << "   <time>" << std::endl;

  outfile << "\t<hour>" << time.hour << "</hour>" << std::endl;
  outfile << "\t<minute>" << time.minute << "</minute>" << std::endl;
  outfile << "\t<second>" << time.second << "</second>" << std::endl;

  outfile << "   </time>" << std::endl;

  // -- primitive stuff -- //
  
  for( std::map<uint,Primitive*>::const_iterator iter = primitives.begin(); iter != primitives.end(); ++iter ){

    uint p = iter->first;

    Primitive* prim = getPrimitivePointer(p);

    RGBAcolor color = prim->getColorRGBA();

    std::string texture_file = prim->getTextureFile();

    std::vector<std::string> pdata = prim->listPrimitiveData();

    if( prim->getType()==PRIMITIVE_TYPE_PATCH ){
      outfile << "   <patch>" << std::endl;
    }else if( prim->getType()==PRIMITIVE_TYPE_TRIANGLE ){
      outfile << "   <triangle>" << std::endl;
    }else if( prim->getType()==PRIMITIVE_TYPE_VOXEL ){
      outfile << "   <voxel>" << std::endl;
    }

    outfile << "\t<UUID>" << p << "</UUID>" << std::endl;
    outfile << "\t<color>" << color.r << " " << color.g << " " << color.b << " " << color.a << "</color>" << std::endl;
    if( prim->hasTexture() ){
      outfile << "\t<texture>" << texture_file << "</texture>" << std::endl;
    }
    if( pdata.size()>0 ){
      for( int i=0; i<pdata.size(); i++ ){
	std::string label = pdata.at(i);
	size_t dsize = prim->getPrimitiveDataSize( label.c_str() );
	HeliosDataType dtype = prim->getPrimitiveDataType( label.c_str() );
	if( dtype==HELIOS_TYPE_UINT ){
	  outfile << "\t<data_uint label=\"" << label << "\">" << std::flush;
	  std::vector<uint> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j) << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_uint>" << std::endl;
	}else if( dtype==HELIOS_TYPE_INT ){
	  outfile << "\t<data_int label=\"" << label << "\">" << std::flush;
	  std::vector<int> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j) << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_int>" << std::endl;
	}else if( dtype==HELIOS_TYPE_FLOAT ){
	  outfile << "\t<data_float label=\"" << label << "\">" << std::flush;
	  std::vector<float> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j) << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_float>" << std::endl;
	}else if( dtype==HELIOS_TYPE_DOUBLE ){
	  outfile << "\t<data_double label=\"" << label << "\">" << std::flush;
	  std::vector<double> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j) << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_double>" << std::endl;
	}else if( dtype==HELIOS_TYPE_VEC2 ){
	  outfile << "\t<data_vec2 label=\"" << label << "\">" << std::flush;
	  std::vector<vec2> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j).x << " " << data.at(j).y << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_vec2>" << std::endl;
	}else if( dtype==HELIOS_TYPE_VEC3 ){
	  outfile << "\t<data_vec3 label=\"" << label << "\">" << std::flush;
	  std::vector<vec3> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_vec3>" << std::endl;
	}else if( dtype==HELIOS_TYPE_VEC4 ){
	  outfile << "\t<data_vec4 label=\"" << label << "\">" << std::flush;
	  std::vector<vec4> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << " " << data.at(j).w << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_vec4>" << std::endl;
	}else if( dtype==HELIOS_TYPE_INT2 ){
	  outfile << "\t<data_int2 label=\"" << label << "\">" << std::flush;
	  std::vector<int2> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j).x << " " << data.at(j).y << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_int2>" << std::endl;
	}else if( dtype==HELIOS_TYPE_INT3 ){
	  outfile << "\t<data_int3 label=\"" << label << "\">" << std::flush;
	  std::vector<int3> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_int3>" << std::endl;
	}else if( dtype==HELIOS_TYPE_INT4 ){
	  outfile << "\t<data_int3 label=\"" << label << "\">" << std::flush;
	  std::vector<int4> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << " " << data.at(j).w << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_int4>" << std::endl;
	}else if( dtype==HELIOS_TYPE_STRING ){
	  outfile << "\t<data_string label=\"" << label << "\">" << std::flush;
	  std::vector<std::string> data;
	  prim->getPrimitiveData( label.c_str(), data );
	  for( int j=0; j<data.size(); j++ ){
	    outfile << data.at(j) << std::flush;
	    if( j!=data.size()-1 ){
	      outfile << " " << std::flush;
	    }
	  }
	  outfile << "</data_string>" << std::endl;
	}
      }
    }

    //Patches
    if( prim->getType()==PRIMITIVE_TYPE_PATCH ){

      Patch* patch = getPatchPointer(p);
      float transform[16];
      prim->getTransformationMatrix(transform);

      outfile << "\t<transform> ";
      for( int i=0; i<16; i++ ){
	outfile << transform[i] << " ";
      }
      outfile << "</transform>" << std::endl;
      std::vector<vec2> uv = patch->getTextureUV();
      if( uv.size()>0 ){
	outfile << "\t<textureUV>" << std::flush;
	for( int i=0; i<uv.size(); i++ ){
	  outfile << uv.at(i).x << " " << uv.at(i).y << std::flush;
	  if( i!=uv.size()-1 ){
	    outfile << " " << std::flush;
	  }
	}
	outfile << "</textureUV>" << std::endl;
      }
      outfile << "   </patch>" << std::endl;

    //Triangles
    }else if( prim->getType()==PRIMITIVE_TYPE_TRIANGLE ){

      float transform[16];
      prim->getTransformationMatrix(transform);

      outfile << "\t<transform> ";
      for( int i=0; i<16; i++ ){
	outfile << transform[i] << " ";
      }
      outfile << "</transform>" << std::endl;

      std::vector<vec2> uv = getTrianglePointer(p)->getTextureUV();
      if( uv.size()>0 ){
	outfile << "\t<textureUV>" << std::flush;
	for( int i=0; i<uv.size(); i++ ){
	  outfile << uv.at(i).x << " " << uv.at(i).y << std::flush;
	  if( i!=uv.size()-1 ){
	    outfile << " " << std::flush;
	  }
	}
	outfile << "</textureUV>" << std::endl;
      }
      outfile << "   </triangle>" << std::endl;

    //Voxels
    }else if( prim->getType()==PRIMITIVE_TYPE_VOXEL ){

      float transform[16];
      prim->getTransformationMatrix(transform);

      outfile << "\t<transform> ";
      for( int i=0; i<16; i++ ){
	outfile << transform[i] << " ";
      }
      outfile << "</transform>" << std::endl;

      outfile << "   </voxel>" << std::endl;
      
    }
    
  }

  // -- global data -- //

  for( std::map<std::string, GlobalData>::const_iterator iter=globaldata.begin(); iter!=globaldata.end(); ++iter ){
    std::string label = iter->first;
    GlobalData data = iter->second;
    HeliosDataType type = data.type;
    if( type==HELIOS_TYPE_UINT ){
      outfile << "   <globaldata_uint label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_uint.at(i) << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_uint>" << std::endl;
    }else if( type==HELIOS_TYPE_INT ){
      outfile << "   <globaldata_int label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_int.at(i) << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_int>" << std::endl;
    }else if( type==HELIOS_TYPE_FLOAT ){
      outfile << "   <globaldata_float label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_float.at(i) << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_float>" << std::endl;
    }else if( type==HELIOS_TYPE_DOUBLE ){
      outfile << "   <globaldata_double label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_double.at(i) << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_double>" << std::endl;
    }else if( type==HELIOS_TYPE_VEC2 ){
      outfile << "   <globaldata_vec2 label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_vec2.at(i).x << " " << data.global_data_vec2.at(i).y << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_vec2>" << std::endl;
    }else if( type==HELIOS_TYPE_VEC3 ){
      outfile << "   <globaldata_vec3 label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_vec3.at(i).x << " " << data.global_data_vec3.at(i).y << " " << data.global_data_vec3.at(i).z << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_vec3>" << std::endl;
    }else if( type==HELIOS_TYPE_VEC4 ){
      outfile << "   <globaldata_vec4 label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_vec4.at(i).x << " " << data.global_data_vec4.at(i).y << " " << data.global_data_vec4.at(i).z << " " << data.global_data_vec4.at(i).w << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_vec4>" << std::endl;
    }else if( type==HELIOS_TYPE_INT2 ){
      outfile << "   <globaldata_int2 label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_int2.at(i).x << " "  << data.global_data_int2.at(i).y << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_int2>" << std::endl;
    }else if( type==HELIOS_TYPE_INT3 ){
      outfile << "   <globaldata_int3 label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_int3.at(i).x << " " << data.global_data_int3.at(i).y << data.global_data_int3.at(i).z << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_int3>" << std::endl;
    }else if( type==HELIOS_TYPE_INT4 ){
      outfile << "   <globaldata_int4 label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_int4.at(i).x << " " << data.global_data_int4.at(i).y << data.global_data_int4.at(i).z << data.global_data_int4.at(i).w << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_int4>" << std::endl;
    }else if( type==HELIOS_TYPE_STRING ){
      outfile << "   <globaldata_string label=\"" << label << "\">" << std::flush;
      for( size_t i=0; i<data.size; i++ ){
	outfile << data.global_data_string.at(i) << std::flush;
	if( i!=data.size-1 ){
	  outfile << " " << std::flush;
	}
      }
      outfile << "</globaldata_string>" << std::endl;
    }

  }

  // -- timeseries -- //

  for( std::map<std::string, std::vector<float> >::const_iterator iter=timeseries_data.begin(); iter!=timeseries_data.end(); ++iter ){

    std::string label = iter->first;

    std::vector<float> data = iter->second;
    std::vector<double> dateval = timeseries_datevalue.at(label);

    assert( data.size()==dateval.size() );

    outfile << "   <timeseries label=\"" << label << "\">" << std::endl;

    for( size_t i=0; i<data.size(); i++ ){

      Date date = queryTimeseriesDate( label.c_str(), i );
      Time time = queryTimeseriesTime( label.c_str(), i );
      
      outfile << "\t<datapoint>" << std::endl;

      outfile << "\t   <date>" << date.day << " " << date.month << " " << date.year << "</date>" << std::endl;

      outfile << "\t   <time>" << time.hour << " " << time.minute << " " << time.second << "</time>" << std::endl;

      outfile << "\t   <value>" << data.at(i) << "</value>" << std::endl;

      outfile << "\t</datapoint>" << std::endl;
    }

    outfile << "   </timeseries>" << std::endl;
    
  }

  // ----------------- //

  outfile << "\n</helios>\n";

  outfile.close();
  
  std::cout << "done." << std::endl;
}

std::vector<uint> Context::loadPLY( const char* filename, const vec3 origin, const float height ){
  return loadPLY( filename, origin, height, make_SphericalCoord(0,0), make_RGBcolor(0,0,1) );
}

std::vector<uint> Context::loadPLY( const char* filename, const vec3 origin, const float height, const SphericalCoord rotation ){
  return loadPLY( filename, origin, height, rotation, make_RGBcolor(0,0,1) );
}

std::vector<uint> Context::loadPLY( const char* filename, const vec3 origin, const float height, const RGBcolor default_color ){
  return loadPLY( filename, origin, height, make_SphericalCoord(0,0), default_color );
}

std::vector<uint> Context::loadPLY( const char* filename, const vec3 origin, const float height, const SphericalCoord rotation, const RGBcolor default_color ){

  std::cout << "Reading PLY file " << filename << "..." << std::flush;

  std::string line, prop;

  uint vertexCount=0, faceCount=0;

  std::vector<vec3> vertices;
  std::vector<std::vector<int> > faces;
  std::vector<RGBcolor> colors;
  std::vector<std::string> properties;

  bool ifColor=false;
  
  std::ifstream inputPly;
  inputPly.open(filename);

  if (!inputPly.is_open()) {
    std::cerr << "Couldn't open " << filename << std::endl;;
    exit(EXIT_FAILURE);
  }

  //--- read header info -----//

  //first line should always be 'ply'
  inputPly>>line;
  if( strcmp("ply",line.c_str())!=0 ){
    std::cerr << "ERROR (loadPLY): " << filename << " is not a PLY file." << std::endl;
    exit(EXIT_FAILURE);
  }

  //read format
  inputPly>>line;
  if( strcmp("format",line.c_str())!=0 ){
    std::cerr << "ERROR (loadPLY): could not determine data format of " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  inputPly>>line;
  if( strcmp("ascii",line.c_str())!=0 ){
    std::cerr << "ERROR (loadPLY): Only ASCII data types are supported." << std::endl;
    exit(EXIT_FAILURE);
  }

  while(strcmp("end_header",line.c_str())!=0){

    inputPly>>line;
    
    if( strcmp("comment",line.c_str())==0 ){
      getline(inputPly, line);
    }
    else if( strcmp("element",line.c_str())==0 ){

      inputPly>>line;

      if( strcmp("vertex",line.c_str())==0 ){
	inputPly>>vertexCount;
      }else if( strcmp("face",line.c_str())==0 ){
	inputPly>>faceCount;
      }

      
    }else if( strcmp("property",line.c_str())==0 ){

      inputPly>>line; //type

      if( strcmp("list",line.c_str())!=0 ){

	inputPly>>prop; //value
	properties.push_back(prop);
	
      }

    }
    
    
  }

  for( uint i=0; i<properties.size(); i++ ){
    if( strcmp(properties.at(i).c_str(),"red")==0 ){
      ifColor = true;
    }
  }
  std::cout<< "forming " << faceCount << " triangles..." << std::flush;

  vertices.resize(vertexCount);
  colors.resize(vertexCount);
  faces.resize(faceCount);


  //--- read vertices ----//

  for( uint row=0; row<vertexCount; row++ ){

    for( uint i=0; i<properties.size(); i++ ){
      if( strcmp(properties.at(i).c_str(),"x")==0 ){ //Note: permuting x,y,z to match our coordinate system (z-vertical instead of y-vertical)
	inputPly >> vertices.at(row).y;
      }else if( strcmp(properties.at(i).c_str(),"y")==0 ){
	inputPly >> vertices.at(row).z;
      }else if( strcmp(properties.at(i).c_str(),"z")==0 ){
	inputPly >> vertices.at(row).x;
      }else if( strcmp(properties.at(i).c_str(),"red")==0 ){
	inputPly >> colors.at(row).r;
	colors.at(row).r /= 255.f;
      }else if( strcmp(properties.at(i).c_str(),"green")==0 ){
	inputPly >> colors.at(row).g;
	colors.at(row).g /= 255.f;
      }else if( strcmp(properties.at(i).c_str(),"blue")==0 ){
	inputPly >> colors.at(row).b;
	colors.at(row).b /= 255.f;
      // }else if( strcmp(properties.at(i).c_str(),"alpha")==0 ){
      // 	inputPly >> colors.at(row).a;
      }else{
	inputPly >> line;
      }
    }

  }

  //determine bounding box

  vec3 boxmin = make_vec3(10000,10000,10000);
  vec3 boxmax = make_vec3(-10000,-10000,-10000);

  for( uint row=0; row<vertexCount; row++ ){

    if(vertices.at(row).x < boxmin.x ){
      boxmin.x = vertices.at(row).x;
    }
    if(vertices.at(row).y < boxmin.y ){
      boxmin.y = vertices.at(row).y;
    }
    if(vertices.at(row).z < boxmin.z ){
      boxmin.z = vertices.at(row).z;
    }

    if(vertices.at(row).x > boxmax.x ){
      boxmax.x = vertices.at(row).x;
    }
    if(vertices.at(row).y > boxmax.y ){
      boxmax.y = vertices.at(row).y;
    }
    if(vertices.at(row).z > boxmax.z ){
      boxmax.z = vertices.at(row).z;
    }    

  }

  //center PLY object at `origin' and scale to have height `height'
  float scl = height/(boxmax.z-boxmin.z);
  for( uint row=0; row<vertexCount; row++ ){
    vertices.at(row).z -= boxmin.z;

    vertices.at(row).x *= scl;
    vertices.at(row).y *= scl;
    vertices.at(row).z *= scl;

    vertices.at(row) = rotatePoint(vertices.at(row),rotation) + origin;
  }

  //--- read faces ----//

  uint v, ID;
  std::vector<uint> UUID;
  for( uint row=0; row<faceCount; row++ ){

    inputPly >> v;

    faces.at(row).resize(v);

    for( uint i=0; i<v; i++ ){
      inputPly >> faces.at(row).at(i);
    }

    //Add triangles to context

    for( uint t=2; t<v; t++ ){ 

      RGBcolor color;
      if( ifColor ){
	color = colors.at(faces.at(row).front());
      }else{
	color = default_color;
      }

      vec3 v0 = vertices.at(faces.at(row).front());
      vec3 v1 = vertices.at(faces.at(row).at(t-1));
      vec3 v2 = vertices.at(faces.at(row).at(t));

      if( (v0-v1).magnitude()==0 || (v0-v2).magnitude()==0 || (v1-v2).magnitude()==0 ){ 
	continue;
      }
      
      ID = addTriangle( v0, v1, v2, color );

      UUID.push_back(ID);

    }
    
  }

  std::cout << "done." << std::endl;

  return UUID;

}

void Context::writePLY( const char* filename ) const{

  std::ofstream PLYfile;
  PLYfile.open(filename);

  PLYfile << "ply" << std::endl << "format ascii 1.0" << std::endl << "comment HELIOS generated" << std::endl;

  std::vector<int3> faces;
  std::vector<vec3> verts;
  std::vector<RGBcolor> colors;

  size_t vertex_count = 0;

  for( std::map<uint,Primitive*>::const_iterator iter = primitives.begin(); iter != primitives.end(); ++iter ){

    uint p = iter->first;

    std::vector<vec3> vertices = getPrimitivePointer(p)->getVertices();
    PrimitiveType type = getPrimitivePointer(p)->getType();
    RGBcolor C = getPrimitivePointer(p)->getColor();
    C.scale(255.f);

    if( type==PRIMITIVE_TYPE_TRIANGLE ){

      faces.push_back( make_int3( vertex_count, vertex_count+1, vertex_count+2 ) );
      for( int i=0; i<3; i++ ){
	verts.push_back( vertices.at(i) );
	colors.push_back( C );
	vertex_count ++;
      }
      
    }else if( type==PRIMITIVE_TYPE_PATCH ){

      faces.push_back( make_int3( vertex_count, vertex_count+1, vertex_count+2 ) );
      faces.push_back( make_int3( vertex_count, vertex_count+2, vertex_count+3 ) );
      for( int i=0; i<4; i++ ){
	verts.push_back( vertices.at(i) );
	colors.push_back( C );
	vertex_count ++;
      }

    }

  }

  PLYfile << "element vertex " << verts.size() << std::endl;
  PLYfile << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
  PLYfile << "property uchar red" << std::endl << "property uchar green" << std::endl << "property uchar blue" << std::endl;
  PLYfile << "element face " << faces.size() << std::endl;
  PLYfile << "property list uchar int vertex_indices" << std::endl << "end_header" << std::endl;

  for( size_t v=0; v<verts.size(); v++ ){
    PLYfile << verts.at(v).x << " " << verts.at(v).y << " " << verts.at(v).z << " " << round(colors.at(v).r) << " " << round(colors.at(v).g) << " " << round(colors.at(v).b) << std::endl;
  }

  for( size_t f=0; f<faces.size(); f++ ){
    PLYfile << "3 " << faces.at(f).x << " " << faces.at(f).y << " " << faces.at(f).z << std::endl; 
  }

  PLYfile.close();

  
}

std::vector<uint> Context::loadOBJ( const char* filename, const vec3 origin, const float height, const SphericalCoord rotation, const RGBcolor default_color ){
  return loadOBJ(filename,origin,height,rotation,default_color,"ZUP");
}
  
std::vector<uint> Context::loadOBJ( const char* filename, const vec3 origin, const float height, const SphericalCoord rotation, const RGBcolor default_color, const char* upaxis ){

  std::cout << "Reading OBJ file " << filename << "..." << std::flush;

  if( strcmp(upaxis,"XUP") && strcmp(upaxis,"YUP") && strcmp(upaxis,"ZUP") ){
    std::cerr << "ERROR (loadOBJ): Up axis of " << upaxis << " is not valid.  Should be one of 'XUP', 'YUP', or 'ZUP'." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string line, prop;

  std::vector<vec3> vertices;
  std::vector<vec2> texture_uv;
  std::map<std::string,std::vector<std::vector<int> > > face_inds, texture_inds;
  //std::vector<RGBcolor> colors;

  std::map<std::string,std::string> material_textures;

  std::vector<uint> UUID;
  
  std::ifstream inputOBJ, inputMTL;
  inputOBJ.open(filename);

  if (!inputOBJ.is_open()) {
    std::cerr << "Couldn't open " << filename << std::endl;;
    exit(EXIT_FAILURE);
  }

  //determine the base file path for 'filename'
  std::string fstring = filename;
  std::string filebase;
  for( int i=fstring.size()-1; i>=0; i-- ){
    if( strncmp(&fstring[i],"/",1)==0 ){
      for( int ii=0; ii<=i; ii++ ){
	filebase.push_back(fstring.at(ii));
      }
      break;
    }
  }

  //determine bounding box
  float boxmin = 100000;
  float boxmax = -100000;

  std::string current_material = "none";

  while( inputOBJ.good() ){

    inputOBJ>>line;

    // ------- COMMENTS --------- //
    if( strcmp("#",line.c_str())==0 ){ 
      getline(inputOBJ, line);
      
    // ------- MATERIAL LIBRARY ------- //
    }else if( strcmp("mtllib",line.c_str())==0 ){ 
      getline(inputOBJ, line);
      std::string material_file = deblank(line.c_str());
      material_textures = loadMTL( filebase, material_file.c_str() );

    // ------- VERTICES --------- //  
    }else if( strcmp("v",line.c_str())==0 ){ 
      getline(inputOBJ, line);
      //parse vertices into points
      vec3 verts(string2vec3(line.c_str()));
      vertices.push_back(verts);

      if(verts.z < boxmin ){
	boxmin = verts.z;
      }
      if(verts.z > boxmax ){
	boxmax = verts.z;
      }    

    // ------- TEXTURE COORDINATES --------- //  
    }else if( strcmp("vt",line.c_str())==0 ){ 
      getline(inputOBJ, line);
      //parse coordinates into uv
      vec2 uv(string2vec2(line.c_str()));
      texture_uv.push_back(uv);

    // ------- MATERIALS --------- //  
    }else if( strcmp("usemtl",line.c_str())==0 ){ 
      getline(inputOBJ, line);
      current_material = line;

    // ------- FACES --------- //  
    }else if( strcmp("f",line.c_str())==0 ){ 
      getline(inputOBJ, line);
      //parse face vertices
      std::istringstream stream(line);
      std::string tmp, digit;
      std::vector<int> f, u;
      while( stream.good() ){

	stream >> tmp;

	digit="";
	int ic = 0;
	for( int i=0; i<tmp.size(); i++ ){
	  if( isdigit(tmp[i]) ){
	    digit.push_back( tmp[i] );
	    ic++;
	  }else{
	    break;
	  }
	}

	f.push_back( atoi(digit.c_str()) );

	digit="";
	for( int i=ic+1; i<tmp.size(); i++ ){
	  if( isdigit(tmp[i]) ){
	    digit.push_back( tmp[i] );
	  }else{
	    break;
	  }
	}
	
	u.push_back( atoi(digit.c_str()) );
	
      }
      face_inds[current_material].push_back(f);
      texture_inds[current_material].push_back(u);

    // ------ OTHER STUFF --------- //    
    }else{
      getline(inputOBJ, line);
    }
  }

  float scl = height/(boxmax-boxmin);

  for(std::map<std::string,std::vector<std::vector<int> > >::const_iterator iter = face_inds.begin(); iter != face_inds.end(); ++iter){

    std::string current_material = iter->first;
    std::string texture;
    if( material_textures.find(current_material)!=material_textures.end() ){
      texture = material_textures.at(current_material);
    }
      
    for( size_t i=0; i<face_inds.at(current_material).size(); i++ ){

      for( uint t=2; t<face_inds.at(current_material).at(i).size(); t++ ){ 
	
	RGBcolor color = default_color;

	vec3 v0 = vertices.at(face_inds.at(current_material).at(i).at(0)-1);
	vec3 v1 = vertices.at(face_inds.at(current_material).at(i).at(t-1)-1);
	vec3 v2 = vertices.at(face_inds.at(current_material).at(i).at(t)-1);

	if( (v0-v1).magnitude()==0 || (v0-v2).magnitude()==0 || (v1-v2).magnitude()==0 ){ 
	  continue;
	}

	if( strcmp(upaxis,"YUP")==0 ){
	  v0 = rotatePointAboutLine(v0,make_vec3(0,0,0),make_vec3(1,0,0),0.5*M_PI);
	  v1 = rotatePointAboutLine(v1,make_vec3(0,0,0),make_vec3(1,0,0),0.5*M_PI);
	  v2 = rotatePointAboutLine(v2,make_vec3(0,0,0),make_vec3(1,0,0),0.5*M_PI);
	}

	v0 = rotatePoint(v0,rotation);
	v1 = rotatePoint(v1,rotation);
	v2 = rotatePoint(v2,rotation);

	uint ID;
	if( !texture.empty() && texture_inds.at(current_material).at(i).size()>0 ){//has texture

	  if( t<texture_inds.at(current_material).at(i).size() ){
	    int iuv0 = texture_inds.at(current_material).at(i).at(0)-1;
	    int iuv1 = texture_inds.at(current_material).at(i).at(t-1)-1;
	    int iuv2 = texture_inds.at(current_material).at(i).at(t)-1;
	   	    
	    ID = addTriangle( origin+v0*scl, origin+v1*scl, origin+v2*scl, texture.c_str(), texture_uv.at(iuv0), texture_uv.at(iuv1), texture_uv.at(iuv2) );

	    vec3 normal = getPrimitivePointer(ID)->getNormal();
	    
	  }
	}else{
	  ID = addTriangle( origin+v0*scl, origin+v1*scl, origin+v2*scl, color );
	}

	UUID.push_back(ID);

      }
    }
  }

  std::cout << "done." << std::endl;

  return UUID;

}

std::map<std::string, std::string> Context::loadMTL( std::string filebase, std::string material_file ){

  std::ifstream inputMTL;

  //first look for mtl file using path given in obj file
  inputMTL.open(material_file.c_str());
  if( !inputMTL.is_open() ){
    //if that doesn't work, try looking in the same directry where obj file is located
    material_file = filebase+material_file;
    material_file.erase( remove( material_file.begin(), material_file.end(), ' ' ), material_file.end() );
    for( int i=material_file.size()-1; i>=0; i-- ){
      if( strcmp(&material_file.at(i),"l")==0 ){
	break;
      }else{
	material_file.erase(material_file.begin()+i);
      }
    }
    if( material_file.empty() ){
      std::cerr << "ERROR (loadMTL): Material file does not have correct file extension (.mtl)." << std::endl;
      exit(EXIT_FAILURE);
    }
    inputMTL.open( material_file.c_str() );
    if( !inputMTL.is_open() ){
      std::cerr << "ERROR (loadOBJ): Material file " << material_file << " given in .obj file cannot be found." << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  std::map<std::string, std::string> material_textures;
    
  std::string line;

  inputMTL>>line;

  while( inputMTL.good() ){

    if( strcmp("#",line.c_str())==0 ){ //comments
      getline(inputMTL, line);
      inputMTL>>line;
    }else if( strcmp("newmtl",line.c_str())==0 ){ //material library
      getline(inputMTL, line);
      std::string material_name = line;
      material_textures[material_name] = "";

      //std::cout << "Found a material library: " << material_name << std::endl;
      
      while( strcmp("newmtl",line.c_str())!=0 && inputMTL.good() ){
	
	if( strcmp("map_a",line.c_str())==0 ){
	  getline(inputMTL, line);
	}else if( strcmp("map_Ka",line.c_str())==0 ){
	  getline(inputMTL, line);
	}else if( strcmp("map_Kd",line.c_str())==0 ){
	  getline(inputMTL, line);

	  std::istringstream stream(line);
	  std::string tmp;
	  while( stream.good() ){
	    stream >> tmp;
	    int e = tmp.size();
	    if( (strncmp(&tmp[e-1],"g",1)==0 && strncmp(&tmp[e-2],"n",1)==0 && strncmp(&tmp[e-3],"p",1)==0 ) || (strncmp(&tmp[e-1],"g",1)==0 && strncmp(&tmp[e-2],"p",1)==0 && strncmp(&tmp[e-3],"j",1)==0 ) || (strncmp(&tmp[e-1],"g",1)==0 && strncmp(&tmp[e-2],"e",1)==0 && strncmp(&tmp[e-3],"p",1)==0  && strncmp(&tmp[e-4],"j",1)==0 ) ){

	      std::string texturefile = tmp;
	      std::ifstream tfile;

	      //first look for texture file using path given in mtl file
	      tfile.open(texturefile.c_str());
	      if( !tfile.is_open() ){
	      	//if that doesn't work, try looking in the same directry where obj file is located
	      	tfile.close();
	      	texturefile = filebase+texturefile;
	      	tfile.open( texturefile.c_str() );
	      	if( !tfile.is_open() ){
	      	  std::cerr << "WARNING (loadOBJ): Texture file " << texturefile << " given in .mtl file cannot be found." << std::endl;
	      	}
	      }
	      tfile.close();

	      material_textures[material_name] = texturefile;
	    }
	  }

	}else if( strcmp("map_Ks",line.c_str())==0 ){
	  getline(inputMTL, line);
	}else{
	  getline(inputMTL, line);
	}

	inputMTL>>line;
      }
      
    }else{
      getline(inputMTL, line);
      inputMTL>>line;
    }
  }

  return material_textures;

}

void Context::writeOBJ( const char* filename ) const{

  //To-Do list for OBJ writer
  // - image files need to be copied to the location where the .obj file is being written otherwise they won't be found.
  // - should parse "filename" to check that extension is .obj, and remove .obj extension when appending .mtl for material file.
  // - it is possible to get duplicate materials. Should have more sophisticated checking for that.
  // - it would make more sense to write patches and alphamasks as quads rather than two triangles

  std::cout << "Writing OBJ file " << filename << "..." << std::flush;

  std::ofstream file;

  file.open(filename);

  file << "# Helios generated OBJ File" << std::endl;
  file << "# baileylab.ucdavis.edu/software/helios" << std::endl;
  file << "mtllib " << filename << ".mtl" << std::endl;

  std::vector<int3> faces;
  std::vector<vec3> verts;
  std::vector<vec2> uv;
  std::vector<int3> uv_inds;

  std::vector<std::string> textures;
  std::vector<RGBcolor> colors;

  size_t vertex_count = 1; //OBJ files start indices at 1
  size_t uv_count = 1;

  for( std::map<uint,Primitive*>::const_iterator iter = primitives.begin(); iter != primitives.end(); ++iter ){

    uint p = iter->first;

    std::vector<vec3> vertices = getPrimitivePointer(p)->getVertices();
    PrimitiveType type = getPrimitivePointer(p)->getType();
    RGBcolor C = getPrimitivePointer(p)->getColor();

    if( type==PRIMITIVE_TYPE_TRIANGLE ){

      faces.push_back( make_int3( vertex_count, vertex_count+1, vertex_count+2 ) );
      colors.push_back( C );
      for( int i=0; i<3; i++ ){
	verts.push_back( vertices.at(i) );
	vertex_count ++;
      }

      std::vector<vec2> uv_v = getTrianglePointer(p)->getTextureUV();
      if( getTrianglePointer(p)->hasTexture() ){
	uv_inds.push_back( make_int3(uv_count, uv_count+1, uv_count+2) );
	textures.push_back(getTrianglePointer(p)->getTextureFile());
	for( int i=0; i<3; i++ ){
	  uv.push_back( uv_v.at(i) );
	  uv_count ++;
	}
      }else{
	textures.push_back("");
	uv_inds.push_back( make_int3(-1,-1,-1) );
      }
	
    }else if( type==PRIMITIVE_TYPE_PATCH ){

      faces.push_back( make_int3( vertex_count, vertex_count+1, vertex_count+2 ) );
      faces.push_back( make_int3( vertex_count, vertex_count+2, vertex_count+3 ) );
      colors.push_back( C );
      colors.push_back( C );
      for( int i=0; i<4; i++ ){
	verts.push_back( vertices.at(i) );
	vertex_count ++;
      }

      std::vector<vec2> uv_v;
      std::string texturefile;
      uv_v = getPatchPointer(p)->getTextureUV();
      texturefile = getPatchPointer(p)->getTextureFile();

      if( getPatchPointer(p)->hasTexture() ){
	textures.push_back(texturefile);
	textures.push_back(texturefile);
	uv_inds.push_back( make_int3(uv_count, uv_count+1, uv_count+2) );
	uv_inds.push_back( make_int3(uv_count, uv_count+2, uv_count+3) );
	if( uv_v.size()==0 ){//default (u,v)
	  uv.push_back( make_vec2(0,1) );
	  uv.push_back( make_vec2(1,1) );
	  uv.push_back( make_vec2(1,0) );
	  uv.push_back( make_vec2(0,0) );
	  uv_count+=4;
	}else{//custom (u,v)
	  for( int i=0; i<4; i++ ){
	    uv.push_back( uv_v.at(i) );
	    uv_count ++;
	  }
	}
      }else{
	textures.push_back("");
	textures.push_back("");
	uv_inds.push_back( make_int3(-1,-1,-1) );
	uv_inds.push_back( make_int3(-1,-1,-1) );
      }

    }

  }

  assert( uv_inds.size()==faces.size() );
  assert( textures.size()==faces.size() );
  assert( colors.size()==faces.size() );

  for( size_t v=0; v<verts.size(); v++ ){
    file << "v " << verts.at(v).x << " " << verts.at(v).y << " " << verts.at(v).z << std::endl; 
  }

  for( size_t v=0; v<uv.size(); v++ ){
    file << "vt " << uv.at(v).x << " " << uv.at(v).y << std::endl;
  }

  std::string current_texture = "";
  int material_count = 1;
  
  for( size_t f=0; f<faces.size(); f++ ){

    if( current_texture.compare(textures.at(f))!=0 ){
      current_texture = textures.at(f);
      file << "usemtl material" << material_count << std::endl;
      material_count ++;
    }
    
    //assert( faces.at(f).x <= verts.size() && faces.at(f).y <= verts.size() && faces.at(f).z <= verts.size() );
    if( uv_inds.at(f).x<0 ){
      file << "f " << faces.at(f).x << " " << faces.at(f).y << " " << faces.at(f).z << std::endl;
    }else{
      //assert( uv_inds.at(f).x <= uv.size() && uv_inds.at(f).y <= uv.size() && uv_inds.at(f).z <= uv.size() );
      file << "f " << faces.at(f).x << "/" << uv_inds.at(f).x << " " << faces.at(f).y << "/" << uv_inds.at(f).y << " " << faces.at(f).z << "/" << uv_inds.at(f).z << std::endl;
    }
  }


  file.close();

  char mtlfilename[50];
  sprintf(mtlfilename,"%s.mtl",filename);
  file.open(mtlfilename);

  current_texture = "";
  material_count = 1;
  RGBcolor current_color = make_RGBcolor(0.010203,0.349302,0.8372910);

  for( size_t f=0; f<faces.size(); f++ ){

    if( textures.at(f).compare("")!=0 ){ //has texture
      if( current_texture.compare(textures.at(f))!=0 ){ //new texture
	current_texture = textures.at(f);
	file << "newmtl material" << material_count << std::endl;
	file << "Ka 1.0 1.0 1.0" << std::endl;
	file << "Kd 1.0 1.0 1.0" << std::endl;
	file << "Ks 0.0 0.0 0.0" << std::endl;
	file << "illum 2 " << std::endl;
	file << "map_Ka " << current_texture << std::endl;
	file << "map_Kd " << current_texture << std::endl;
	file << "map_d " << current_texture << std::endl;
	
	material_count ++;
      }
    }else{ //does not have texture
      if( current_color.r!=colors.at(f).r && current_color.g!=colors.at(f).g && current_color.b!=colors.at(f).b ){ //new color
	current_texture = textures.at(f);
	current_color = colors.at(f);
	file << "newmtl material" << material_count << std::endl;
	file << "Ka " << current_color.r << " " << current_color.g << " " << current_color.b << std::endl;
	file << "Kd " << current_color.r << " " << current_color.g << " " << current_color.b << std::endl;
	file << "Ks 0.0 0.0 0.0" << std::endl;
	file << "illum 2 " << std::endl;
	
	material_count ++;
      }
    }

  }
  
  file.close();
	    

  std::cout << "done." << std::endl;
  
}

Context::~Context(){

  for(  std::map<uint,Primitive*>::iterator iter = primitives.begin(); iter != primitives.end(); ++iter ){
    Primitive* prim = getPrimitivePointer(iter->first);
    delete prim;
  }

}
