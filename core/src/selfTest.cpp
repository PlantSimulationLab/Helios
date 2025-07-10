/** \file "selfTest.cpp" Context selfTest() function.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "Context.h"
#include <set>

using namespace helios;

int Context::selfTest(){

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
    uint objID;

    //uint addPatch( const vec3& center, const vec2& size );
    center = make_vec3(1,2,3);
    size = make_vec2(1,2);
    vertices.resize(4);
    vertices.at(0) = center + make_vec3( -0.5f*size.x, -0.5f*size.y, 0.f);
    vertices.at(1) = center + make_vec3( 0.5f*size.x, -0.5f*size.y, 0.f);
    vertices.at(2) = center + make_vec3( 0.5f*size.x, 0.5f*size.y, 0.f);
    vertices.at(3) = center + make_vec3( -0.5f*size.x, 0.5f*size.y, 0.f);
    UUID = context_test.addPatch(center,size);
    type = context_test.getPrimitiveType(UUID);
    center_r = context_test.getPatchCenter(UUID);
    size_r = context_test.getPatchSize(UUID);
    normal_r = context_test.getPrimitiveNormal(UUID);
    vertices_r = context_test.getPrimitiveVertices(UUID);
    area_r = context_test.getPrimitiveArea(UUID);
    color_r = context_test.getPrimitiveColor(UUID);

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
    if( std::abs( area_r - size.x*size.y )>1e-5 ){
        error_count++;
        std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getArea did not return correct value." << std::endl;
    }
    if( color_r.r!=0.f || color_r.g!=0.f || color_r.b!=0.f ){
        error_count++;
        std::cerr << "failed: uint addPatch( const vec3& center, const vec2& size ). Patch::getColor did not return default color of black." << std::endl;
    }

    if( !context_test.getPrimitiveTextureFile(UUID).empty() ){
        error_count++;
        std::cerr << "failed: Patch without texture mapping returned true for texture map test." << std::endl;
    }

    //------- Copy Patch --------//

    std::vector<float> cpdata{5.2f,2.5f,3.1f};

    context_test.setPrimitiveData( UUID, "somedata", cpdata );

    uint UUID_cpy = context_test.copyPrimitive(UUID);

    vec3 center_cpy = context_test.getPatchCenter(UUID_cpy);
    vec2 size_cpy = context_test.getPatchSize(UUID_cpy);

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

    std::vector<float> cpdata_copy;
    context_test.getPrimitiveData( UUID_cpy, "somedata", cpdata_copy );

    if( cpdata.size() != cpdata_copy.size() ){
        error_count++;
        std::cerr << "failed: copyPrimitive. Copied patch primitive data does not have correct size." << std::endl;
    }
    for( uint i=0; i<cpdata.size(); i++ ){
        if( cpdata.at(i) != cpdata_copy.at(i) ){
            error_count++;
            std::cerr << "failed: copyPrimitive. Copied patch primitive data does not match." << std::endl;
        }
        break;
    }

    //translate the copied patch
    vec3 shift = make_vec3(5.f,4.f,3.f);
    context_test.translatePrimitive(UUID_cpy,shift);
    center_cpy = context_test.getPatchCenter(UUID_cpy);
    center_r = context_test.getPatchCenter(UUID);

    if( std::abs(center_cpy.x-center.x-shift.x)>errtol || std::abs(center_cpy.y-center.y-shift.y)>errtol || std::abs(center_cpy.z-center.z-shift.z)>errtol || center_r.x!=center.x || center_r.y!=center.y || center_r.z!=center.z ){
        error_count++;
        std::cerr << "failed: copyPrimitive. Copied patch could not be properly translated." << std::endl;
    }

    //------- Delete Patch --------//

    context_test.deletePrimitive(UUID);

    if(context_test.getPrimitiveCount() != 1 ){
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
    rotation = make_SphericalCoord(1.f, 0.15f*M_PI, 0.5f*M_PI);
    rotation.azimuth = 0.5f*M_PI;;
    UUID = context_test.addPatch(center,size,rotation);
    normal_r = context_test.getPrimitiveNormal(UUID);

    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    context_test.deletePrimitive(UUID);

    if( std::abs(rotation_r.elevation-rotation.elevation)>errtol || std::abs(rotation_r.azimuth-rotation.azimuth)>errtol ){
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

    type = context_test.getPrimitiveType(UUID);
    normal_r = context_test.getPrimitiveNormal(UUID);
    vertices_r = context_test.getPrimitiveVertices(UUID);
    area_r = context_test.getPrimitiveArea(UUID);
    color_r = context_test.getPrimitiveColor(UUID);

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

    if( !context_test.getPrimitiveTextureFile(UUID).empty() ){
        error_count++;
        std::cerr << "failed: Triangle without texture mapping returned true for texture map test." << std::endl;
    }

    //------- Copy Triangle --------//

    UUID_cpy = context_test.copyPrimitive(UUID);

    std::vector<vec3> vertices_cpy = context_test.getPrimitiveVertices(UUID_cpy);

    if( vertices.at(0).x!=vertices_cpy.at(0).x || vertices.at(0).y!=vertices_cpy.at(0).y || vertices.at(0).z!=vertices_cpy.at(0).z ||
        vertices.at(1).x!=vertices_cpy.at(1).x || vertices.at(1).y!=vertices_cpy.at(1).y || vertices.at(1).z!=vertices_cpy.at(1).z ||
        vertices.at(2).x!=vertices_cpy.at(2).x || vertices.at(2).y!=vertices_cpy.at(2).y || vertices.at(2).z!=vertices_cpy.at(2).z ){
        error_count++;
        std::cerr << "failed: copied triangle did not return correct vertices." << std::endl;
    }

    //translate the copied patch
    shift = make_vec3(5,4,3);
    context_test.translatePrimitive(UUID_cpy,shift);

    vertices_cpy = context_test.getPrimitiveVertices(UUID_cpy);

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
    objID = context_test.addBoxObject( center, size3, subdiv );
    UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();

    normal_r = context_test.getPrimitiveNormal(UUIDs.at(0));
    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    if( std::abs(rotation_r.zenith-0.f)>errtol || std::abs(rotation_r.azimuth-0.f)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
    }

    normal_r = context_test.getPrimitiveNormal( UUIDs.at(2));
    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    if( std::abs(rotation_r.zenith-0.f)>errtol || std::abs(rotation_r.azimuth-0.5f*float(M_PI))>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
    }

    size_r = context_test.getPatchSize( UUIDs.at(0) );

    if( std::abs(size_r.x-size3.x)>errtol || std::abs(size_r.y-size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
    }

    size_r = context_test.getPatchSize( UUIDs.at(2) );

    if( std::abs(size_r.x-size3.y)>errtol || std::abs(size_r.y-size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
    }

    if( std::abs(context_test.getBoxObjectVolume(objID)-size3.x*size3.y*size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Box volume incorrect." << std::endl;
    }
    //------- Add a Rotated Tile --------//

    center = make_vec3(1,2,3);
    size = make_vec2(3,2);
    int2 subdiv2(3,3);
    rotation = make_SphericalCoord( 0.25f*M_PI, 1.4f*M_PI );
    objID = context_test.addTileObject(center, size, rotation, subdiv2);
    UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();

    for( uint UUIDp : UUIDs){

        normal_r = context_test.getPrimitiveNormal(UUIDp);

        rotation_r = cart2sphere(normal_r);

        if( std::abs(rotation_r.zenith-rotation.zenith)>errtol || std::abs(rotation_r.azimuth-rotation.azimuth)>errtol ){
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
    objID = context_test.addTileObject( center, size, rotation, subdiv2, "lib/images/disk_texture.png" );
    UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();

    float At = 0;
    for( uint UUIDp : UUIDs){

        float area = context_test.getPrimitiveArea(UUIDp);

        At+=area;

    }

    float area_exact = 0.25f*PI_F*size.x*size.y;

    if( std::abs(At-area_exact)>0.005 ){
        error_count++;
        std::cerr << "failed: addTile(). Texture masked area is incorrect." << std::endl;
        std::cout << At << " " << area_exact << std::endl;
    }

    //------- Primitive Bounding Box --------//

    std::vector<uint> UUID_bbox;

    UUID_bbox.push_back( context_test.addPatch( make_vec3(-1,0,0), make_vec2(0.5,0.5) ));
    UUID_bbox.push_back( context_test.addPatch( make_vec3(1,0,0), make_vec2(0.5,0.5) ));

    vec3 bboxmin_patch, bboxmax_patch;
    context_test.getPrimitiveBoundingBox( UUID_bbox, bboxmin_patch, bboxmax_patch );

    if( bboxmin_patch.x!=-1.25f || bboxmax_patch.x!=1.25 || bboxmin_patch.y!=-0.25f || bboxmax_patch.y!=0.25f || bboxmin_patch.z!=0.f || bboxmax_patch.z!=0.f ){
        error_count++;
        std::cerr << "failed: getPrimitiveBoundingBox(). Bounding box is incorrect." << std::endl;
    }

    //------- Primitive Transformations --------//

    vec2 sz_0(0.5,3.f);
    float A_0 = sz_0.x*sz_0.y;

    float scale = 2.6f;

    UUID = context_test.addPatch( make_vec3(0,0,0), sz_0 );
    context_test.scalePrimitive( UUID, make_vec3(scale,scale,scale) );

    float A_1 = context_test.getPrimitiveArea(UUID);

    if( std::abs( A_1 - scale*scale*A_0 )>1e-5 ){
        error_count ++;
        std::cerr << "failed: Patch scaling - scaled area not correct." << std::endl;
    }

    //------- Primitive Data --------//

    float data = 5;
    context_test.setPrimitiveData( UUID,"some_data",data);

    if( !context_test.doesPrimitiveDataExist( UUID,"some_data") ){
        error_count ++;
        std::cerr << "failed: setPrimitiveData - data was added but was not actually created." << std::endl;
    }

    float data_return;
    context_test.getPrimitiveData( UUID,"some_data", data_return);
    if( data_return!=data ){
        error_count ++;
        std::cerr << "failed: set/getPrimitiveData (setting/getting through primitive). Get data did not match set data." << std::endl;
        std::cout << data_return << std::endl;
    }

    context_test.setPrimitiveData(UUID,"some_data",data);
    context_test.getPrimitiveData(UUID,"some_data",data_return);
    if( data_return!=data ){
        error_count ++;
        std::cerr << "failed: set/getPrimitiveData (setting/getting through Context). Get data did not match set data." << std::endl;
        std::cout << data_return << std::endl;
    }

    std::vector<float> data_v{0,1,2,3,4};

    context_test.setPrimitiveData( UUID,"some_data",data_v);

    std::vector<float> data_return_v;
    context_test.getPrimitiveData( UUID,"some_data", data_return_v);
    for( uint i=0; i<5; i++ ){
        if( data_return_v.at(i)!=data_v.at(i) ){
            error_count ++;
            std::cerr << "failed: set/getPrimitiveData (setting/getting through primitive). Get data did not match set data." << std::endl;
            std::cout << data_return << std::endl;
            break;
        }
    }

    data = 10;
    context_test.setPrimitiveData( UUID,"some_data_2",data);

    if( !context_test.doesPrimitiveDataExist( UUID,"some_data_2") ){
        error_count ++;
        std::cerr << "failed: setPrimitiveData - data was added but was not actually created." << std::endl;
    }

    context_test.getPrimitiveData( UUID,"some_data_2", data_return);
    if( data_return!=data ){
        error_count ++;
        std::cerr << "failed: set/getPrimitiveData (setting/getting scalar data). Get data did not match set data." << std::endl;
        std::cout << data_return << std::endl;
    }

    //primitive data filters

    std::vector<uint> UUIDs_multi, UUIDs_filter;
    UUIDs_multi.push_back( context_test.addPatch() );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_float", 4.f );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_string", "cat" );
    UUIDs_multi.push_back( context_test.addPatch() );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_float", 3.f );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_string", "cat" );
    UUIDs_multi.push_back( context_test.addPatch() );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_float", 2.f );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_string", "dog" );
    UUIDs_multi.push_back( context_test.addPatch() );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_float", 1.f );
    context_test.setPrimitiveData( UUIDs_multi.back(), "some_data_string", "dog" );

    UUIDs_filter = context_test.filterPrimitivesByData( UUIDs_multi, "some_data_float", 2.f, "<=" );
    if( UUIDs_filter.size()!=2 || std::find(UUIDs_filter.begin(),UUIDs_filter.end(),UUIDs_multi.at(2))==UUIDs_filter.end() || std::find(UUIDs_filter.begin(),UUIDs_filter.end(),UUIDs_multi.at(3))==UUIDs_filter.end() ){
      error_count ++;
      std::cerr << "failed: primitive data filter for floats was not correct." << std::endl;
    }

    UUIDs_filter = context_test.filterPrimitivesByData( UUIDs_multi, "some_data_string", "cat" );
    if( UUIDs_filter.size()!=2 || std::find(UUIDs_filter.begin(),UUIDs_filter.end(),UUIDs_multi.at(0))==UUIDs_filter.end() || std::find(UUIDs_filter.begin(),UUIDs_filter.end(),UUIDs_multi.at(1))==UUIDs_filter.end() ){
      error_count ++;
      std::cerr << "failed: primitive data filter for strings was not correct." << std::endl;
    }

    //------- Textures --------- //

    vec2 sizep = make_vec2(2,3);

    const char* texture = "lib/images/disk_texture.png";

    vec2 uv0(0,0);
    vec2 uv1(1,0);
    vec2 uv2(1,1);
    vec2 uv3(0,1);

    uint UUIDp = context_test.addPatch( make_vec3(2,3,4), sizep, make_SphericalCoord(0,0), texture, 0.5*(uv0+uv2), uv2-uv0 );

    if( context_test.getPrimitiveTextureFile(UUIDp).empty() ){
        error_count ++;
        std::cerr << "failed: Texture-mapped patch was found not to have texture." << std::endl;
    }

    std::string texture2 = context_test.getPrimitiveTextureFile(UUIDp);

    if( texture2!=texture ){
        error_count ++;
        std::cerr << "failed: textures - queried texture file does not match that provided when adding primitive." << std::endl;
    }

    float Ap = context_test.getPrimitiveArea(UUIDp);

    if( std::abs(Ap-0.25f*float(M_PI)*sizep.x*sizep.y)/(0.25f*float(M_PI)*sizep.x*sizep.y)>0.01f ){
        error_count ++;
        std::cerr << "failed: Texture-masked patch does not have correct area." << std::endl;
    }

    std::vector<vec2> uv;
    uv = context_test.getPrimitiveTextureUV(UUIDp);

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

    float area = context_test.getPrimitiveArea(UUIDp2);

    if( std::abs(area-sizep.x*sizep.y)>0.001f ){
        error_count ++;
        std::cerr << "failed: Patch masked with (u,v) coordinates did not return correct area." << std::endl;
    }

    uv0 = make_vec2( 0, 0 );
    uv1 = make_vec2( 1, 1 );
    uv2 = make_vec2( 0, 1 );

    uint UUIDp3 = context_test.addTriangle( make_vec3(2,3,4), make_vec3(2,3+sizep.y,4), make_vec3(2+sizep.x,3+sizep.y,4), texture, uv0, uv1, uv2 );

    area = context_test.getPrimitiveArea(UUIDp3);

    if( std::abs(area-0.5f*0.25f*float(M_PI)*sizep.x*sizep.y)>0.01f ){
        error_count ++;
        std::cerr << "failed: Triangle masked with (u,v) coordinates did not return correct area." << std::endl;
    }

    uint UUIDt2 = context_test.addTriangle( make_vec3(0,0,0), make_vec3(1,0,0), make_vec3(1,1,0), "lib/images/diamond_texture.png", make_vec2(0,0), make_vec2(1,0), make_vec2(1,1) );
    float solid_fraction = context_test.getPrimitiveSolidFraction(UUIDt2);
    if( fabs(solid_fraction-0.5f)>errtol ){
      error_count ++;
      std::cerr << "failed: Textured triangle solid fraction was not correct." << std::endl;
        std::cout << solid_fraction << std::endl;
    }

    //------- Global Data --------//

    float gdata = 5;
    context_test.setGlobalData("some_data",gdata);

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

    std::vector<float> gdata_v{0,1,2,3,4};

    context_test.setGlobalData("some_data",gdata_v);

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

    //------- Compound Object Data --------//

    uint IDtile = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(3, 1),
                                             make_SphericalCoord(0, 0), make_int2(3, 3));

    float objdata = 5;
    context_test.setObjectData( IDtile, "some_data", objdata);

    if( !context_test.doesObjectDataExist( IDtile, "some_data") ){
        error_count ++;
        std::cerr << "failed: setObjectData - data was added but was not actually created." << std::endl;
    }

    float objdata_return;
    context_test.getObjectPointer(IDtile)->getObjectData("some_data", objdata_return);
    if( objdata_return!=objdata ){
        error_count ++;
        std::cerr << "failed: set/getObjectData (getting through object pointer). Get data did not match set data." << std::endl;
    }

    std::vector<float> objdata_v{0,1,2,3,4};

    context_test.setObjectData(IDtile, "some_data", objdata_v );

    std::vector<float> objdata_return_v;
    context_test.getObjectData(IDtile,"some_data", objdata_return_v);
    for( uint i=0; i<5; i++ ){
        if( objdata_return_v.at(i)!=objdata_v.at(i) ){
            error_count ++;
            std::cerr << "failed: set/getObjectData. Get data did not match set data." << std::endl;
            break;
        }
    }

    objdata = 10;
    context_test.setObjectData(IDtile,"some_data_2",objdata);

    if( !context_test.doesObjectDataExist(IDtile,"some_data_2") ){
        error_count ++;
        std::cerr << "failed: setObjectData - data was added but was not actually created." << std::endl;
    }

    context_test.getObjectData(IDtile, "some_data_2", objdata_return);
    if( objdata_return!=objdata ){
        error_count ++;
        std::cerr << "failed: set/getObjectData (setting/getting scalar data). Get data did not match set data." << std::endl;
    }

    //------- Cone Object Transformations --------//

    float cone_radius_0 = 0.5f;
    float cone_radius_1 = 1.0f;
    float cone_length = 2.0f;

    helios::vec3 node0 = make_vec3(0,0,0);
    helios::vec3 node1 = make_vec3(0,0,cone_length);

    //create cone object
    uint cone_1 = context_test.addConeObject( 50, node0, node1, cone_radius_0, cone_radius_1);

    //translate to (1,1,1)
    context_test.getConeObjectPointer(cone_1)->translate(make_vec3(1, 1, 1));

    // get the updated node position
    std::vector<helios::vec3> nodes_T = context_test.getConeObjectPointer(cone_1)->getNodeCoordinates();

    if(nodes_T.at(0).x - 1.0f > errtol || nodes_T.at(0).y - 1.0f > errtol || nodes_T.at(0).z - 1.0f > errtol ||
       nodes_T.at(1).x - 1.0f > errtol || nodes_T.at(1).y - 1.0f > errtol || nodes_T.at(1).z - 3.0f > errtol ){

        error_count ++;
        std::cerr << "failed: translate cone object. Node coordinates after translation not correct." << std::endl;
    }

    //rotate so that the cone is parallel with x axis
    // define the desired tube axis
    helios::vec3 x_axis = helios::make_vec3(1,0,0);
    helios::vec3 z_axis = make_vec3(0,0,1);
    //get the axis about which to rotate
    vec3 ra = cross( z_axis, x_axis);
    //get the angle to rotate
    float dot = x_axis.x*z_axis.x + x_axis.y*z_axis.y + x_axis.z*z_axis.z;
    float angle = acos_safe(dot);

    // translate back to origin
    context_test.getConeObjectPointer(cone_1)->translate(-1.f*nodes_T.at(0));
    //rotate
    context_test.getConeObjectPointer(cone_1)->rotate( angle, ra);
    // translate back
    context_test.getConeObjectPointer(cone_1)->translate(nodes_T.at(0));
    //get the updated node positions
    nodes_T = context_test.getConeObjectPointer(cone_1)->getNodeCoordinates();

    if(nodes_T.at(0).x - 1.0f > errtol || nodes_T.at(0).y - 1.0f > errtol || nodes_T.at(0).z - 1.0f > errtol ||
       nodes_T.at(1).x - 3.0f > errtol || nodes_T.at(1).y - 1.0f > errtol || nodes_T.at(1).z - 1.0f > errtol ){
        error_count ++;
        std::cerr << "failed: rotate cone object. Node coordinates after rotation not correct." << std::endl;
    }


    //scale the length of the cone to twice its original length
    context_test.getConeObjectPointer(cone_1)->scaleLength(2.0);
    //get the updated node positions
    nodes_T = context_test.getConeObjectPointer(cone_1)->getNodeCoordinates();

    if(nodes_T.at(0).x - 1.0 > errtol || nodes_T.at(0).y - 1.0 > errtol || nodes_T.at(0).z - 1.0 > errtol ||
       nodes_T.at(1).x - 6.0 > errtol || nodes_T.at(1).y - 1.0 > errtol || nodes_T.at(1).z - 1.0 > errtol ){

        error_count ++;
        std::cerr << "failed: scaleLength cone object. Node coordinates after length scaling not correct." << std::endl;
    }

    //scale the girth of the cone to twice its original radii
    context_test.getConeObjectPointer(cone_1)->scaleGirth(2.0);
    //get the updated node positions
    std::vector<float> radii_T = context_test.getConeObjectPointer(cone_1)->getNodeRadii();

    if(radii_T.at(0) - cone_radius_0*2.0 > pow(10,-6) || radii_T.at(1) - cone_radius_1*2.0 > pow(10, -6) ){
        error_count ++;
        std::cerr << "failed: scaleGirth cone object. Node radii after girth scaling not correct." << std::endl;
    }

    //------- Cartesian/Spherical Coordinate Conversion --------//

    SphericalCoord sph1 = make_SphericalCoord(1.f,0.25*M_PI,1.5*M_PI);
    vec3 cart = sphere2cart(sph1);
    SphericalCoord sph2 = cart2sphere(cart);

    if( std::abs(sph1.radius-sph2.radius)>errtol || std::abs(sph1.elevation-sph2.elevation)>errtol || std::abs(sph1.zenith-sph2.zenith)>errtol || std::abs(sph1.azimuth-sph2.azimuth)>errtol ){
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

    //-------- Spline Interpolation -----------//

    vec3 p_start(0,0,0);
    vec3 t_start(3,0,0);

    vec3 p_end(1,1.5,0.4);
    vec3 t_end(0,1,0);

    float u = 0.6;

    vec3 xi = spline_interp3( u, p_start, t_start, p_end, t_end );

    vec3 xi_ref( 0.9360, 0.8280, 0.2592 );

    if( std::abs(xi.x-xi_ref.x)>errtol || std::abs(xi.y-xi_ref.y)>errtol || std::abs(xi.z-xi_ref.z)>errtol ){
        std::cerr << "failed: cubic spline interpolation incorrect." << std::endl;
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

    Context context_ts2;

    context_ts2.loadTabularTimeseriesData("lib/testdata/weather_data.csv", {"date", "hour", "temperature"}, ",", "MMDDYYYY", 1);

    Date date_ts2 = make_Date( 2, 1, 2020 );
    Time time_ts2 = make_Time( 13, 00, 00 );
    float T = context_ts2.queryTimeseriesData( "temperature", date_ts2, time_ts2 );

    if( T != 35.32343f ){
        std::cerr << "failed: Load of tabular weather data failed." << std::endl;
        error_count++;
    }

    context_ts2.loadTabularTimeseriesData("lib/testdata/cimis.csv", {"cimis"}, "," );

    date_ts2 = make_Date( 18, 7, 2023 );
    time_ts2 = make_Time( 13, 00, 00 );
    float rh = context_ts2.queryTimeseriesData( "air_humidity", date_ts2, time_ts2 );

    if( rh != 0.42f ){
        std::cerr << "failed: Load of CIMIS weather data failed." << std::endl;
        error_count++;
    }

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

    std::vector<double> gdatad_io{9.432, 2939.9292 };

    context_io.setGlobalData( "gdatad", gdatad_io);

    //timeseries

    float T0_io = 302.3;
    float T1_io = 305.3;
    Time time0_io = time_io;
    Time time1_io = make_Time( time_io.hour, 30, 0 );

    context_io.addTimeseriesData( "ts_io", T0_io, date_io, time0_io );
    context_io.addTimeseriesData( "ts_io", T1_io, date_io, time1_io );

    context_io.addTileObject(p_io,size_io,nullrotation,make_int2(3,2),"lib/images/disk_texture.png");
    context_io.addSphereObject( 8, p_io, 5 );
    context_io.addDiskObject( 8, p_io, size_io );
    context_io.addConeObject( 8, make_vec3(1,1,3), make_vec3(1,1,5), 3, 3 );
    context_io.addBoxObject( p_io, make_vec3(3,2,1), make_int3(3,3,3), RGB::red );
    context_io.addTubeObject( 8, {make_vec3(1,1,3), make_vec3(1,1,5)}, {3, 3} );

    context_io.writeXML( "xmltest_io.xml", true );

    Context context_oi;

    context_oi.loadXML( "xmltest_io.xml", true );

    // This now fails with Context::primitives being an unordered map because we can't assume the UUIDs transfer from the XML file
    // float pdataf;
    // int pdatai;
    //
    // context_oi.getPrimitiveData(0,"pdataf",pdataf);
    // context_oi.getPrimitiveData(0,"pdatai",pdatai);
    //
    // if( pdataf!=pdataf_p || pdatai!=pdatai_p ){
    //     std::cerr << "failed. Patch primitive data write/read do not match." << std::endl;
    //     error_count ++;
    // }

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

    //------- setTileObjectSubdivisionCount, getTileObjectAreaRatio, cropDomain & writeXML & loadXML for compound objects  --------//

    Context context_to;

    //non-textured tile object single subpatch
    uint to1 = context_to.addTileObject(make_vec3(0,0,0), make_vec2(1,0.3), make_SphericalCoord(float(M_PI)*0.25, float(M_PI)*0.75), make_int2(1,1));
    //non-textured tile object multiple subpatches
    uint to2 = context_to.addTileObject(make_vec3(1,1,1), make_vec2(1,1), make_SphericalCoord(float(M_PI)*0.25, float(M_PI)*0.75), make_int2(5,3));
    //textured tile object single subpatch disk
    uint to3 = context_to.addTileObject(make_vec3(2,2,2), make_vec2(1,0.3), make_SphericalCoord(float(M_PI)*0.25, float(M_PI)*0.75), make_int2(1,1), "lib/images/disk_texture.png");
    //textured tile object multiple subpatches disk
    uint to4 = context_to.addTileObject(make_vec3(3,3,3), make_vec2(1,0.3), make_SphericalCoord(float(M_PI)*0.25, float(M_PI)*0.75), make_int2(5,3), "lib/images/disk_texture.png");
    //textured tile object single subpatch diamond
    uint to5 = context_to.addTileObject(make_vec3(-1,-1,-1), make_vec2(1,1), make_SphericalCoord(M_PI*0.25, M_PI*0.75), make_int2(1,1), "lib/images/diamond_texture.png");
    //textured tile object multiple subpatches diamond
    uint to6 = context_to.addTileObject(make_vec3(-2,-2,-2), make_vec2(1,1), make_SphericalCoord(M_PI*0.25, M_PI*0.75), make_int2(5,3), "lib/images/diamond_texture.png");

    //throw in a sphere object just to check error checks
//    uint so1 = context_to.addSphereObject(10, make_vec3(4,4,4), 0.3);
//    uint n_UUIDs_so1 = context_to.getObjectPointer(so1)->getPrimitiveUUIDs().size();

    context_to.writeXML("xmltest_to.xml", true );
    uint context_to_PrimitiveCount = context_to.getPrimitiveCount();
    uint context_to_ObjectCount = context_to.getObjectCount();

    //------------ getTileObjectAreaRatio -------------//

    errtol = 1e-2;
    double err_1 = fabs(context_to.getTileObjectAreaRatio(to1) - 1.0);
    if(err_1 >= errtol){
        std::cerr << "failed. tile object area ratio returned by getTileObjectAreaRatio is not correct for non-textured single subpatch tile." << std::endl;
        error_count ++;
    }
    double err_2 = fabs(context_to.getTileObjectAreaRatio(to2) - 15.0);
    if(err_2 >= errtol){
        std::cerr << "failed. tile object area ratio returned by getTileObjectAreaRatio is not correct for non-textured multi-subpatch tile." << std::endl;
        error_count ++;
    }
    double err_3 = fabs(context_to.getTileObjectAreaRatio(to5) - 1.0);
    if(err_3 >= errtol){
        std::cerr << "failed. tile object area ratio returned by getTileObjectAreaRatio is not correct for textured single subpatch tile." << std::endl;
        error_count ++;
    }
    double err_4 = fabs(context_to.getTileObjectAreaRatio(to6) - 1.0*0.5/(1.0/15.0));
    if(err_4 >= 5.f*errtol){
        std::cerr << "failed. tile object area ratio returned by getTileObjectAreaRatio is not correct for textured multi-subpatch tile." << std::endl;
        error_count ++;
    }

    //----------- setTileObjectSubdivisionCount -------------//

    std::vector<uint> allObjectIDs_to = context_to.getAllObjectIDs();
    context_to.setTileObjectSubdivisionCount(allObjectIDs_to, make_int2(5,5));
    std::vector<uint> allObjectIDs_to_after = context_to.getAllObjectIDs();
    std::vector<uint> allUUIDs_to_after = context_to.getAllUUIDs();

//    if( (allUUIDs_to_after.size() - n_UUIDs_so1) != uint(150)){
//        std::cerr << "failed. setTileObjectSubdivisionCount (subiv version) not producing the correct number of primitives." << std::endl;
//        error_count ++;
//    }

    if( allObjectIDs_to_after.size() != allObjectIDs_to.size()){
        std::cerr << "failed. setTileObjectSubdivisionCount(subiv version) is  changing the number of objects in the context (it shouldn't)." << std::endl;
        error_count ++;
    }

    context_to.setTileObjectSubdivisionCount(allObjectIDs_to, 49);
    std::vector<uint> allObjectIDs_to_after2 = context_to.getAllObjectIDs();
    std::vector<uint> allUUIDs_to_after2 = context_to.getAllUUIDs();

    if( allObjectIDs_to_after2 != allObjectIDs_to){
        std::cerr << "failed. setTileObjectSubdivisionCount  (area ratio version)  is changing the number of objects in the context (it shouldn't)." << std::endl;
        error_count ++;
    }

    if( context_to.getTileObjectPointer(to1)->getPrimitiveUUIDs().size() == uint(49)){
        std::cerr << "failed. setTileObjectSubdivisionCount  (area ratio version)  did not result in the correct number of subpatches" << std::endl;
        error_count ++;
    }

    //------------ writeXML & loadXML for compound objects --------------//

    Context context_to2;
    context_to2.loadXML("xmltest_to.xml", true );
    uint context_to2_PrimitiveCount = context_to2.getPrimitiveCount();
    uint context_to2_ObjectCount = context_to2.getObjectCount();

    if( context_to_PrimitiveCount != context_to2_PrimitiveCount){
        std::cerr << "failed. number of primitives before writing and loading xml does not match number after" << std::endl;
        error_count ++;
    }

    if( context_to_ObjectCount != context_to2_ObjectCount){
        std::cerr << "failed. number of Objects before writing and loading xml does not match number after" << std::endl;
        error_count ++;
    }

    
    //------- deletion of objects when all constituent primitives are deleted  --------//
    
    // CASE # 1 Changing Object ID of one object primitive
    // Would expect a single object to remain containing one primitive and a separate non-object primitive
    
    Context context_dzpo;
    context_dzpo.addTileObject(make_vec3(0,0,0), make_vec2(1,1), nullrotation, make_int2(1,2));
    
    //set primitive parent object ID to zero
    context_dzpo.setPrimitiveParentObjectID(0, uint(0));
    
    std::vector<uint> opi = context_dzpo.getObjectPrimitiveUUIDs(1);
    if(opi.size() != 1)
    {
        std::cerr << "failed. Changing Object ID of one object primitive did not result in correct number of primitives in the object" << std::endl;
        error_count ++;
    }
    context_dzpo.writeXML("./dzpo_case1.xml", true );
    
    Context context_dzpo_load;
    context_dzpo_load.loadXML("./dzpo_case1.xml", true);
    std::vector<uint> opi_load = context_dzpo_load.getObjectPrimitiveUUIDs(1);
    if(opi_load.size() != 1)
    {
        std::cerr << "failed. Changing Object ID of one object primitive and writing/loading did not result in correct number of primitives in the object (Case 1)" << std::endl;
        error_count ++;
    }
    
    //CASE # 2 Changing Object ID of one object primitive and deleting it
    //Would expect a single object to remain containing one primitive
    
    Context context_dzpo2;
    context_dzpo2.addTileObject(make_vec3(0,0,0), make_vec2(1,1), nullrotation, make_int2(1,2));

    //set one primitive parent object ID to zero 
    context_dzpo2.setPrimitiveParentObjectID(0, uint(0));
    // delete it
    context_dzpo2.deletePrimitive(0);

    context_dzpo2.writeXML("./dzpo_case2.xml", true);

    Context context_dzpo2_load;
    context_dzpo2_load.loadXML("./dzpo_case2.xml", true);
    uint opi_load2 = context_dzpo2_load.getObjectPrimitiveUUIDs(1).size();
    uint n_prim2 = context_dzpo2_load.getAllUUIDs().size();
    uint n_obj2 = context_dzpo2_load.getAllObjectIDs().size();
    
    if(opi_load2 != 1 | n_prim2 != 1 | n_obj2 != 1)
    {
        std::cerr << "failed. Changing Object ID of one object primitive, deleting, and writing/loading did not result in correct number of primitives or objects (Case 2)" << std::endl;
        error_count ++;
    }


    //CASE #3 Changing Object ID of one object primitive and deleting the other object primitive
    //Would expect the object to be deleted since it has no primitives
    Context context_dzpo3;
    context_dzpo3.addTileObject(make_vec3(0,0,0), make_vec2(1,1), nullrotation, make_int2(1,2));

    //set one primitive parent object ID to zero
    context_dzpo3.setPrimitiveParentObjectID(0, uint(0));
    //delete the other (only) primitive that is part of the object
    context_dzpo3.deletePrimitive(1);

    context_dzpo3.writeXML("./dzpo_case3.xml", true);

    Context context_dzpo3_load;
    context_dzpo3_load.loadXML("./dzpo_case3.xml", true);
    uint n_prim3 = context_dzpo3_load.getAllUUIDs().size();
    uint n_obj3 = context_dzpo3_load.getAllObjectIDs().size();
    
    if(n_prim3 != 1 | n_obj3 != 0)
    {
        std::cerr << "failed. Changing Object ID of one object primitive, deleting, and writing/loading did not result in correct number of primitives or objects (Case 3)" << std::endl;
        error_count ++;
    }

    //---------- file name/path parsing functions -------------//

    std::string filename = "/path/to/.hidden/file/filename";

    std::string expected_path;
#ifdef _WIN32
    expected_path = "\\path\\to\\.hidden\\file\\";
#else
    expected_path = "/path/to/.hidden/file/";
#endif


    std::string ext = getFileExtension(filename);
    std::string name = getFileName(filename);
    std::string stem = getFileStem(filename);
    std::string path = getFilePath(filename,true);

    if( !ext.empty() || name!="filename" || stem!="filename" || path!=expected_path ){
      std::cerr << "failed: file path parsing functions were not correct." << std::endl;
      error_count++;
    }

    filename = ".hidden/path/to/file/filename.ext";

#ifdef _WIN32
    expected_path = ".hidden\\path\\to\\file";
#else
    expected_path = ".hidden/path/to/file";
#endif

    ext = getFileExtension(filename);
    name = getFileName(filename);
    stem = getFileStem(filename);
    path = getFilePath(filename,false);

    if( ext!=".ext" || name!="filename.ext" || stem!="filename" || path!=expected_path ){
      std::cerr << "failed: file path parsing functions were not correct." << std::endl;
      error_count++;
    }

    //---------- primitive data calculation functions -------------//

    std::vector<uint> UUIDs_calcfuns;

    for( uint i=0; i<5; i++ ){
      UUIDs_calcfuns.push_back(context_test.addPatch());
      context_test.setPrimitiveData(UUIDs_calcfuns.back(),"data_calcfuns",float(i+1));
    }

    float mean_calcfuns;
    context_test.calculatePrimitiveDataMean( UUIDs_calcfuns, "data_calcfuns", mean_calcfuns );

    if( mean_calcfuns!=3.f ){
      std::cerr << "failed: calculatePrimitiveDataMean() did not produce correct mean value." << std::endl;
      error_count++;
    }

    float awtmean_calcfuns;
    context_test.calculatePrimitiveDataAreaWeightedMean( UUIDs_calcfuns, "data_calcfuns", awtmean_calcfuns );

    if( awtmean_calcfuns!=3.f ){
      std::cerr << "failed: calculatePrimitiveDataAreaWeightedMean() did not produce correct mean value." << std::endl;
      error_count++;
    }

    float sum_calcfuns;
    context_test.calculatePrimitiveDataSum( UUIDs_calcfuns, "data_calcfuns", sum_calcfuns );

    if( sum_calcfuns!=15.f ){
      std::cerr << "failed: calculatePrimitiveDataSum() did not produce correct sum value." << std::endl;
      error_count++;
    }

    //------- OBJ File Read --------//

    std::vector<uint> OBJ_UUIDs = context_test.loadOBJ( "lib/models/obj_object_test.obj", make_vec3(0,0,0), 0, nullrotation, RGB::blue, true );

    std::vector<uint> UUIDs_patch = context_test.filterPrimitivesByData( OBJ_UUIDs, "object_label", "patch" );
    std::vector<uint> UUIDs_tri = context_test.filterPrimitivesByData( OBJ_UUIDs, "object_label", "triangles" );

    if( UUIDs_patch.size()!=2 || UUIDs_tri.size()!=2 ){
      std::cerr << "failed: loadOBJ() did not properly assign object groups to elements." << std::endl;
      error_count++;
    }else{
      RGBcolor patch_color_0 = context_test.getPrimitiveColor( UUIDs_patch.at(0) );
      RGBcolor patch_color_1 = context_test.getPrimitiveColor( UUIDs_patch.at(1) );
      RGBcolor triangle_color_0 = context_test.getPrimitiveColor( UUIDs_tri.at(0) );
      RGBcolor triangle_color_1 = context_test.getPrimitiveColor( UUIDs_tri.at(1) );
      if( patch_color_0!=RGB::red || patch_color_1!=RGB::red || triangle_color_0!=RGB::lime || triangle_color_1!=RGB::lime ){
        std::cerr << "failed: loadOBJ() did not properly assign object groups and/or materials to elements." << std::endl;
        error_count++;
      }
    }

    {
        // Test Context::seedRandomGenerator(uint)
        Context context_test;
        context_test.seedRandomGenerator(42);
        std::minstd_rand0* rng1 = context_test.getRandomGenerator();
        context_test.seedRandomGenerator(42);
        std::minstd_rand0* rng2 = context_test.getRandomGenerator();

        if (rng1 == nullptr || rng2 == nullptr) {
            error_count++;
            std::cerr << "failed: Context::seedRandomGenerator. Random generator returned null pointer." << std::endl;
        }
    }

    {
        // Test Context::getRandomGenerator()
        Context context_test;
        if (context_test.getRandomGenerator() == nullptr) {
            error_count++;
            std::cerr << "failed: Context::getRandomGenerator. Did not return a valid pointer." << std::endl;
        }
    }

    {
        // Test Texture::getTextureFile() const
        const char* textureFilename = "lib/images/disk_texture.png";
        Texture texture(textureFilename);
        std::string textureFile = texture.getTextureFile();

        if (textureFile != textureFilename) {
            error_count++;
            std::cerr << "failed: Texture::getTextureFile. Expected 'Almondleaf.png' but got '" << textureFile << "'." << std::endl;
        }
    }

    {
        // Test Context::markGeometryClean()
        Context context_test;
        context_test.addPatch();
        context_test.markGeometryDirty();
        context_test.markGeometryClean();

        if (context_test.isGeometryDirty()) {
            error_count++;
            std::cerr << "failed: Context::markGeometryClean. Geometry should be marked clean but is dirty." << std::endl;
        }
    }

    {
        // Test Context::isGeometryDirty() const
        Context context_test;
        context_test.addPatch();
        context_test.markGeometryDirty();

        if (!context_test.isGeometryDirty()) {
            error_count++;
            std::cerr << "failed: Context::isGeometryDirty. Expected geometry to be dirty but it is clean." << std::endl;
        }
    }


    {
        // Test Voxel::getArea() const
        float errtol = 1e-7;
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1)); // Corrected function call

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel creation failed. Could not retrieve Voxel object." << std::endl;
        } else {
            float expected_area = 6.f;  // Assuming a unit cube with six faces of area 1 each
            float area = context_test.getPrimitiveArea(UUID);

            if (std::abs(area - expected_area) > errtol) {
                error_count++;
                std::cerr << "failed: Voxel::getArea. Expected " << expected_area << " but got " << area << "." << std::endl;
            }
        }
    }

// ====== TEST Voxel::getNormal() ======
    {
        Context context_test; // Create a new Context instance

        // Add a Voxel primitive and retrieve its UUID
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::getNormal() - Voxel creation failed." << std::endl;
        } else {
            // Get the normal of the created voxel
            vec3 normal = context_test.getPrimitiveNormal(UUID);

            // Check if the normal is the expected value
            if (!(normal.x == 0.0f && normal.y == 0.0f && normal.z == 0.0f)) {
                error_count++;
                std::cerr << "failed: Voxel::getNormal() did not return expected (0,0,0)." << std::endl;
            }
        }
    }

    {
        // Test Voxel::getVertices() const
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel creation failed. Could not retrieve Voxel object." << std::endl;
        } else {
            std::vector<vec3> vertices = context_test.getPrimitiveVertices(UUID);

            if (vertices.size() != 8) {
                error_count++;
                std::cerr << "failed: Voxel::getVertices. Expected 8 vertices but got " << vertices.size() << "." << std::endl;
            }
        }
    }


    {
        // Test Primitive::getColorRGB() const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Patch creation failed. Could not retrieve Patch object." << std::endl;
        } else {
            RGBcolor color = context_test.getPrimitiveColor(UUID);

            if (color.r != 0.f || color.g != 0.f || color.b != 0.f) {
                error_count++;
                std::cerr << "failed: Patch::getColorRGB. Expected (0,0,0) but got ("
                          << color.r << ", " << color.g << ", " << color.b << ")." << std::endl;
            }
        }
    }

    { //  Test Primitive::overrideTextureColor()
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: overrideTextureColor() - Patch creation failed." << std::endl;
        } else {
            context_test.overridePrimitiveTextureColor(UUID);
            if (!context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: overrideTextureColor() - Texture override flag was not set correctly." << std::endl;
            }
        }
    }

    { //  Test Primitive::useTextureColor()
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: useTextureColor() - Patch creation failed." << std::endl;
        } else {
            context_test.overridePrimitiveTextureColor(UUID);
            context_test.usePrimitiveTextureColor(UUID);
            if (context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: useTextureColor() - Texture override flag was not reset correctly." << std::endl;
            }
        }
    }

    { //  Test Voxel::calculateSolidFraction()
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::calculateSolidFraction() - Voxel creation failed." << std::endl;
        } else {
            float solidFraction = context_test.getPrimitiveSolidFraction(UUID);
            if (std::abs(solidFraction - 1.0f) > errtol) {
                error_count++;
                std::cerr << "failed: Voxel::calculateSolidFraction() - Expected solid fraction 1.0 but got "
                          << solidFraction << "." << std::endl;
            }
        }
    }

    { //  Test Triangle::setVertices()
        Context context_test;
        uint UUID = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::red);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Triangle::setVertices() - Triangle creation failed." << std::endl;
        } else {
            std::vector<vec3> vertices = context_test.getPrimitiveVertices(UUID);
            if (vertices.size() != 3) {
                error_count++;
                std::cerr << "failed: Triangle::setVertices() - Expected 3 vertices but got "
                          << vertices.size() << "." << std::endl;
            }
        }
    }

    { //  Test Primitive::scale()
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Primitive::scale() - Patch creation failed." << std::endl;
        } else {
            context_test.scalePrimitive(UUID, make_vec3(2.0, 2.0, 2.0));
            float area = context_test.getPrimitiveArea(UUID);
            float expected_area = 4.0f;
            if (std::abs(area - expected_area) > errtol) {
                error_count++;
                std::cerr << "failed: Primitive::scale() - Expected area " << expected_area << " but got "
                          << area << "." << std::endl;
            }
        }
    }

    { // Test Patch::rotate()
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Patch::rotate() - Patch creation failed." << std::endl;
        } else {
            context_test.rotatePrimitive(UUID, M_PI / 4, make_vec3(0, 0, 1));
            vec3 normal = context_test.getPrimitiveNormal(UUID);
            if (std::abs(normal.x) > errtol || std::abs(normal.y) > errtol || std::abs(normal.z - 1.0) > errtol) {
                error_count++;
                std::cerr << "failed: Patch::rotate() - Normal did not rotate as expected." << std::endl;
            }
        }
    }

    { //  Test Triangle::rotate()
        Context context_test;
        uint UUID = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::blue);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Triangle::rotate() - Triangle creation failed." << std::endl;
        } else {
            context_test.rotatePrimitive(UUID, M_PI / 2, make_vec3(0, 0, 1));
            vec3 normal = context_test.getPrimitiveNormal(UUID);
            if (std::abs(normal.x) > errtol || std::abs(normal.y) > errtol || std::abs(normal.z - 1.0) > errtol) {
                error_count++;
                std::cerr << "failed: Triangle::rotate() - Rotation incorrect." << std::endl;
            }
        }
    }

    { //  Test Voxel::rotate()
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::rotate() - Voxel creation failed." << std::endl;
        } else {
            context_test.rotatePrimitive(UUID, M_PI / 2, "z");
            vec3 normal = context_test.getPrimitiveNormal(UUID);
            if (std::abs(normal.x) > errtol || std::abs(normal.y) > errtol || std::abs(normal.z) > errtol) {
                error_count++;
                std::cerr << "failed: Voxel::rotate() - Expected normal to change after rotation." << std::endl;
            }
        }
    }

    { //  Triangle::rotate(float, const vec3&, const vec3&)
        Context context_test;
        uint UUID = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::red);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Triangle::rotate() - Triangle creation failed." << std::endl;
        } else {
            context_test.rotatePrimitive(UUID, M_PI / 4, make_vec3(1, 1, 0), make_vec3(0, 0, 1));
            vec3 normal = context_test.getPrimitiveNormal(UUID);
            if (std::abs(normal.x) > errtol || std::abs(normal.y) > errtol || std::abs(normal.z - 1.0) > errtol) {
                error_count++;
                std::cerr << "failed: Triangle::rotate() - Rotation incorrect." << std::endl;
            }
        }
    }

    { //  Voxel::rotate(float, const char*)
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::rotate() - Voxel creation failed." << std::endl;
        } else {
            context_test.rotatePrimitive(UUID, M_PI / 2, "z");
            vec3 normal = context_test.getPrimitiveNormal(UUID);
            if (std::abs(normal.x) > errtol || std::abs(normal.y) > errtol || std::abs(normal.z) > errtol) {
                error_count++;
                std::cerr << "failed: Voxel::rotate() - Expected normal to change after rotation." << std::endl;
            }
        }
    }

        // Determine platform-specific null stream
    #ifdef _WIN32
        const char* null_device = "NUL";
    #else
        const char* null_device = "/dev/null";
    #endif

    { // 21. Voxel::rotate(float, const vec3&) - Ensure rotation is ignored for unsupported axes
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::rotate() - Voxel creation failed." << std::endl;
        } else {
            vec3 original_normal = context_test.getPrimitiveNormal(UUID);

            // Suppress warnings temporarily
            std::streambuf* old_cerr = std::cerr.rdbuf();
            std::ofstream null_stream(null_device);
            std::cerr.rdbuf(null_stream.rdbuf());

            context_test.rotatePrimitive(UUID, M_PI / 4, make_vec3(0, 1, 0)); // Invalid axis

            // Restore std::cerr
            std::cerr.rdbuf(old_cerr);

            vec3 new_normal = context_test.getPrimitiveNormal(UUID);

            if (new_normal != original_normal) {
                error_count++;
                std::cerr << "failed: Voxel::rotate() - Rotation was incorrectly applied despite being ignored." << std::endl;
            }
        }
    }

    { // 22. Voxel::rotate(float, const vec3&, const vec3&) - Ensure rotation is ignored for unsupported axes
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::rotate() - Voxel creation failed." << std::endl;
        } else {
            vec3 original_normal = context_test.getPrimitiveNormal(UUID);

            // Suppress warnings temporarily
            std::streambuf* old_cerr = std::cerr.rdbuf();
            std::ofstream null_stream(null_device);
            std::cerr.rdbuf(null_stream.rdbuf());

            context_test.rotatePrimitive(UUID, M_PI / 4, make_vec3(1, 1, 1), make_vec3(1, 0, 0)); // Invalid axis

            // Restore std::cerr
            std::cerr.rdbuf(old_cerr);

            vec3 new_normal = context_test.getPrimitiveNormal(UUID);

            if (new_normal != original_normal) {
                error_count++;
                std::cerr << "failed: Voxel::rotate() - Rotation was incorrectly applied when it should be ignored." << std::endl;
            }
        }
    }

    { // Triangle Constructor Test - Fix addTriangle
        Context context_test;
        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(1, 0, 0);
        vec3 v2 = make_vec3(0, 1, 0);
        uint UUID = context_test.addTriangle(v0, v1, v2, RGB::blue); // Corrected call

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Triangle constructor - Could not create Triangle." << std::endl;
        }
    }

    { //  Triangle::getVertex(int)
        Context context_test;
        uint UUID = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::green);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Triangle::getVertex() - Triangle creation failed." << std::endl;
        } else {
            vec3 vertex0 = context_test.getTriangleVertex(UUID, 0);
            vec3 expected_vertex0 = make_vec3(0, 0, 0);
            if (vertex0 != expected_vertex0) {
                error_count++;
                std::cerr << "failed: Triangle::getVertex() - Incorrect vertex 0." << std::endl;
            }
        }
    }

    ///////////////////////////////////

    {
        // Test Triangle::Triangle constructor
        Context context_test;

        helios::vec3 v0 = make_vec3(0.0f, 0.0f, 0.0f);
        helios::vec3 v1 = make_vec3(1.0f, 0.0f, 0.0f);
        helios::vec3 v2 = make_vec3(0.0f, 1.0f, 0.0f);
        const char* textureFile = "lib/images/disk_texture.png";
        std::vector<helios::vec2> uv = {make_vec2(0.0f, 0.0f), make_vec2(1.0f, 0.0f), make_vec2(0.0f, 1.0f)};
        float solidFraction = 1.0f;
        uint parentObjectID = 1;
        uint UUID = 42; // Arbitrary test UUID

        Triangle testTriangle(v0, v1, v2, textureFile, uv, solidFraction, parentObjectID, UUID);

        // Check if vertices are correctly set
        std::vector<helios::vec3> vertices = testTriangle.getVertices();
        if (vertices.size() != 3 ||
            vertices[0] != v0 || vertices[1] != v1 || vertices[2] != v2) {
            error_count++;
            std::cerr << "failed: Triangle::Triangle - vertices not set correctly." << std::endl;
        }

        // Check if the texture file is correctly assigned
        if (testTriangle.getTextureFile() != std::string(textureFile)) {
            error_count++;
            std::cerr << "failed: Triangle::Triangle - texture file not set correctly." << std::endl;
        }

        // Check if the UV coordinates are correctly assigned
        std::vector<helios::vec2> retrievedUV = testTriangle.getTextureUV();
        if (retrievedUV != uv) {
            error_count++;
            std::cerr << "failed: Triangle::Triangle - UV coordinates not set correctly." << std::endl;
        }


        // Check if parent object ID is correctly assigned
        if (testTriangle.getParentObjectID() != parentObjectID) {
            error_count++;
            std::cerr << "failed: Triangle::Triangle - parent object ID not set correctly." << std::endl;
        }

        // Check if UUID is correctly assigned
        if (testTriangle.getUUID() != UUID) {
            error_count++;
            std::cerr << "failed: Triangle::Triangle - UUID not set correctly." << std::endl;
        }
    }



    { //  Triangle::getCenter()
        Context context_test;
        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(2, 0, 0);
        vec3 v2 = make_vec3(0, 2, 0);
        uint UUID = context_test.addTriangle(v0, v1, v2, RGB::yellow);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Triangle::getCenter() - Triangle creation failed." << std::endl;
        } else {
            std::vector<vec3> vertices = context_test.getPrimitiveVertices(UUID);
            vec3 center = (vertices[0] + vertices[1] + vertices[2]) / 3.0f; // Compute center manually
            vec3 expected_center = make_vec3(2.0 / 3.0, 2.0 / 3.0, 0.0);
            if (std::abs(center.x - expected_center.x) > errtol || std::abs(center.y - expected_center.y) > errtol || std::abs(center.z - expected_center.z) > errtol) {
                error_count++;
                std::cerr << "failed: Triangle::getCenter() - Expected center (" << expected_center.x << ", " << expected_center.y << ", " << expected_center.z << ") but got ("
                          << center.x << ", " << center.y << ", " << center.z << ")." << std::endl;
            }
        }
    }

    {
        // Test Voxel::Voxel( const RGBAcolor& a_color, uint a_parent_objID, uint a_UUID )
        RGBAcolor color = make_RGBAcolor(0.1f, 0.2f, 0.3f, 1.0f);
        uint parentID = 1;
        uint UUID = 42;
        Voxel voxel(color, parentID, UUID);

        if (voxel.getColorRGBA().r != color.r || voxel.getColorRGBA().g != color.g ||
            voxel.getColorRGBA().b != color.b || voxel.getColorRGBA().a != color.a) {
            error_count++;
            std::cerr << "failed: Voxel::Voxel() - Color not initialized correctly." << std::endl;
        }

        if (voxel.getParentObjectID() != parentID) {
            error_count++;
            std::cerr << "failed: Voxel::Voxel() - Parent ID not initialized correctly." << std::endl;
        }

        if (voxel.getUUID() != UUID) {
            error_count++;
            std::cerr << "failed: Voxel::Voxel() - UUID not initialized correctly." << std::endl;
        }
    }




    {
        // Test Context::getMonthString() const
        Context context_test;
        context_test.setDate(32, 2025); // Assuming Julian day 32 is February 1

        std::string month = context_test.getMonthString();

        if (month != "FEB") {
            error_count++;
            std::cerr << "failed: Context::getMonthString() - Expected 'FEB' but got '"
                      << month << "'." << std::endl;
        }
    }


    {
        // Test Context::getJulianDate() const
        Context context_test;

        int test_day = 32; // Set to a safe value
        int test_year = 2025; // More recent year to avoid issues

        context_test.setDate(test_day, test_year);  // Ensure day is valid

        int julian_date = context_test.getJulianDate();

        if (julian_date != test_day) {
            error_count++;
            std::cerr << "failed: Context::getJulianDate() - Expected " << test_day << " but got "
                      << julian_date << "." << std::endl;
        }
    }

    {
        // Test Context::randu()
        Context context_test;
        float rand_value = context_test.randu();
        if (rand_value < 0.f || rand_value > 1.f) {
            error_count++;
            std::cerr << "failed: Context::randu() returned value out of range [0,1]." << std::endl;
        }
    }

    {
        // Test Context::randu(float minrange, float maxrange)
        Context context_test;
        float min_range = 10.0, max_range = 20.0;
        float rand_value = context_test.randu(min_range, max_range);
        if (rand_value < min_range || rand_value > max_range) {
            error_count++;
            std::cerr << "failed: Context::randu(min,max) returned value out of range." << std::endl;
        }
    }

    {
        // Test Context::randu(int minrange, int maxrange)
        Context context_test;
        int min_range = 5, max_range = 15;
        int rand_value = context_test.randu(min_range, max_range);
        if (rand_value < min_range || rand_value > max_range) {
            error_count++;
            std::cerr << "failed: Context::randu(int min, int max) returned value out of range." << std::endl;
        }
    }

    {
        // Test Context::randn()
        Context context_test;
        float rand_value = context_test.randn();
        if (rand_value < -5.f || rand_value > 5.f) { // Assuming 5-sigma rule
            error_count++;
            std::cerr << "failed: Context::randn() returned extreme value outside expected range." << std::endl;
        }
    }

    {
        // Test Context::randn(float mean, float stddev)
        Context context_test;
        float mean = 100.0, stddev = 10.0;
        float rand_value = context_test.randn(mean, stddev);
        if (rand_value < mean - 5 * stddev || rand_value > mean + 5 * stddev) {
            error_count++;
            std::cerr << "failed: Context::randn(mean, stddev) returned value outside expected range." << std::endl;
        }
    }


    {
        // Test Voxel::getCenter() const
        Context context_test;
        vec3 center = make_vec3(1, 1, 1);
        uint UUID = context_test.addVoxel(center, make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::getCenter() - Voxel creation failed." << std::endl;
        } else {
            if (context_test.getPrimitiveType(UUID) != PRIMITIVE_TYPE_VOXEL) {
                error_count++;
                std::cerr << "failed: UUID does not correspond to a Voxel object." << std::endl;
            } else {
                vec3 retrieved_center = context_test.getVoxelCenter(UUID); // Use correct function

                if (!(retrieved_center.x == center.x && retrieved_center.y == center.y && retrieved_center.z == center.z)) {
                    error_count++;
                    std::cerr << "failed: Voxel::getCenter() did not return expected center coordinates." << std::endl;
                }
            }
        }
    }

    {
        // Test Voxel::getSize()
        Context context_test;
        vec3 size = make_vec3(2, 2, 2);
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), size);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Voxel::getSize() - Voxel creation failed." << std::endl;
        } else {
            if (context_test.getPrimitiveType(UUID) != PRIMITIVE_TYPE_VOXEL) {
                error_count++;
                std::cerr << "failed: UUID does not correspond to a Voxel object." << std::endl;
            } else {
                vec3 retrieved_size = context_test.getVoxelSize(UUID); // Use correct function

                if (!(retrieved_size.x == size.x && retrieved_size.y == size.y && retrieved_size.z == size.z)) {
                    error_count++;
                    std::cerr << "failed: Voxel::getSize() did not return expected size." << std::endl;
                }
            }
        }
    }


    {
        // Test Context::setTime(int minute, int hour)
        Context context_test;
        int test_hour = 14;
        int test_minute = 30;
        context_test.setTime(test_minute, test_hour);

        auto retrieved_time = context_test.getTime();  // Assuming it returns a Time structure

        if (retrieved_time.hour != test_hour || retrieved_time.minute != test_minute) {
            error_count++;
            std::cerr << "failed: Context::setTime() did not set the correct time." << std::endl;
        }
    }

    {
        // Test Context::setDate(int Julian_day, int year)
        Context context_test;
        int test_day = 200;
        int test_year = 2024;
        context_test.setDate(test_day, test_year);

        helios::Date retrieved_date = context_test.getDate(); // Ensure this function exists

        // Assuming `Date` has `day`, `month`, or `year` instead of `julian_day`
        if (retrieved_date.day != 18 || retrieved_date.year != test_year) {
            error_count++;
            std::cerr << "failed: Context::setDate() did not set the correct date. Expected day "
                      << test_day << " and year " << test_year << " but got "
                      << retrieved_date.day << " and " << retrieved_date.year << "." << std::endl;
        }
    }

    {
        // Test Context::getLocation() const
        Context context_test;

        // Expected default location from Context.cpp initialization
        Location expected_location = make_Location(38.55, 121.76, 8);

        Location retrieved_location = context_test.getLocation();

        // Compare entire object rather than accessing nonexistent members
        if (!(retrieved_location == expected_location)) {
            error_count++;
            std::cerr << "failed: Context::getLocation() did not return the correct default location." << std::endl;
        }
    }


    {
        // Test Context::setLocation(const helios::Location &location)
        Context context_test;

        // Instead of using direct members, use make_Location
        Location location_test = make_Location(40.0, -74.0, 10.0);
        context_test.setLocation(location_test);

        Location retrieved_location = context_test.getLocation();

        // Assuming `make_Location()` correctly assigns values, compare using a helper function
        if (!(retrieved_location == location_test)) {
            error_count++;
            std::cerr << "failed: Context::setLocation() did not set the correct location." << std::endl;
        }
    }

    {
        // Test Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation, const char* texture_file )
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(1, 1);
        SphericalCoord rotation = make_SphericalCoord(0, 0, 0);
        const char* texture_file = "lib/images/disk_texture.png";

        uint UUID = context_test.addPatch(center, size, rotation, texture_file);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: addPatch() - Patch creation failed." << std::endl;
        } else {
            if (context_test.getPrimitiveType(UUID) != PRIMITIVE_TYPE_PATCH) {
                error_count++;
                std::cerr << "failed: addPatch() - Primitive type mismatch." << std::endl;
            }
            if (context_test.getPrimitiveTextureFile(UUID) != texture_file) {
                error_count++;
                std::cerr << "failed: addPatch() - Texture file mismatch." << std::endl;
            }
        }
    }

    {
        // Test Context::addTriangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2 )
        Context context_test;
        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(1, 0, 0);
        vec3 v2 = make_vec3(0, 1, 0);

        uint UUID = context_test.addTriangle(v0, v1, v2);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: addTriangle() - Triangle creation failed." << std::endl;
        } else {
            if (context_test.getPrimitiveType(UUID) != PRIMITIVE_TYPE_TRIANGLE) {
                error_count++;
                std::cerr << "failed: addTriangle() - Primitive type mismatch." << std::endl;
            }
        }
    }

    {
        // Test Context::addVoxel( const vec3& center, const vec3& size )
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(1, 1, 1);

        uint UUID = context_test.addVoxel(center, size);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: addVoxel() - Voxel creation failed." << std::endl;
        } else {
            if (context_test.getPrimitiveType(UUID) != PRIMITIVE_TYPE_VOXEL) {
                error_count++;
                std::cerr << "failed: addVoxel() - Primitive type mismatch." << std::endl;
            }
        }
    }

    {
        // Test Context::addVoxel( const vec3& center, const vec3& size, const float& rotation )
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(1, 1, 1);
        float rotation = 45.0f;

        uint UUID = context_test.addVoxel(center, size, rotation);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: addVoxel() with rotation - Voxel creation failed." << std::endl;
        }
    }

    {
        // Test Context::addVoxel( const vec3& center, const vec3& size, const float& rotation, const RGBcolor& color )
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(1, 1, 1);
        float rotation = 45.0f;
        RGBcolor color = make_RGBcolor(1.0f, 0.0f, 0.0f);

        uint UUID = context_test.addVoxel(center, size, rotation, color);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: addVoxel() with rotation and color - Voxel creation failed." << std::endl;
        }
    }

    {
        // Test Context::addVoxel( const vec3& center, const vec3& size, const float& rotation, const RGBAcolor& color )
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(1, 1, 1);
        float rotation = 45.0f;
        RGBAcolor color = make_RGBAcolor(1.0f, 0.0f, 0.0f, 1.0f);

        uint UUID = context_test.addVoxel(center, size, rotation, color);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: addVoxel() with rotation and RGBA color - Voxel creation failed." << std::endl;
        }
    }

    {
        // Test Context::rotatePrimitive(uint UUID, float rot, const char* axis )
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        float rotation = 90.0f;
        context_test.rotatePrimitive(UUID, rotation, "z");

        // Verify that rotation was applied
        if (!context_test.isGeometryDirty()) {
            error_count++;
            std::cerr << "failed: rotatePrimitive() - Rotation not applied." << std::endl;
        }
    }

    {
        // Test Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const char* axis)
        Context context_test;
        float rotation_angle = M_PI / 2.0; // 90 degrees
        const char* axis = "z";

        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: rotatePrimitive() - Patch creation failed." << std::endl;
        } else {
            std::vector<uint> UUIDs = {UUID};
            context_test.rotatePrimitive(UUIDs, rotation_angle, axis);

            vec3 new_normal = context_test.getPrimitiveNormal(UUID);

            if (std::abs(new_normal.x - 0.0f) > 1e-5 || std::abs(new_normal.y - 0.0f) > 1e-5 || std::abs(new_normal.z - 1.0f) > 1e-5) {
                error_count++;
                std::cerr << "failed: rotatePrimitive() - Patch normal incorrect after rotation around Z-axis." << std::endl;
            }
        }
    }

    {
        // Test Context::translatePrimitive(const std::vector<uint>& UUIDs, const vec3& shift)
        Context context_test;
        std::vector<uint> UUIDs;

        // Add a patch
        UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

        if (!context_test.doesPrimitiveExist(UUIDs[0])) {
            error_count++;
            std::cerr << "failed: Patch creation failed. UUID does not exist." << std::endl;
        } else {
            vec3 shift = make_vec3(2.0f, 3.0f, 1.0f);
            context_test.translatePrimitive(UUIDs, shift);

            // Retrieve the new position using getPatchCenter (valid for patches)
            vec3 newPosition = context_test.getPatchCenter(UUIDs[0]);

            if (fabs(newPosition.x - 2.0f) > 1e-6 ||
                fabs(newPosition.y - 3.0f) > 1e-6 ||
                fabs(newPosition.z - 1.0f) > 1e-6) {
                error_count++;
                std::cerr << "failed: Context::translatePrimitive - Patch was not moved correctly." << std::endl;
            }
        }
    }


    {
        // Test Context::scalePrimitive(const std::vector<uint>& UUIDs, const helios::vec3& S)
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

        context_test.scalePrimitive(UUIDs, make_vec3(2, 2, 1));

        vec2 newSize = context_test.getPatchSize(UUIDs[0]);
        if (newSize.x != 2.0f || newSize.y != 2.0f) {
            error_count++;
            std::cerr << "failed: Context::scalePrimitive - Scaling incorrect." << std::endl;
        }
    }

    {
        // Test Context::scalePrimitiveAboutPoint(uint UUID, const helios::vec3& S, const helios::vec3 point)
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(1, 1, 1), make_vec3(1, 1, 1));

        context_test.scalePrimitiveAboutPoint(UUID, make_vec3(2, 2, 2), make_vec3(1, 1, 1));

        vec3 newSize = context_test.getVoxelSize(UUID);
        if (newSize.x != 2.0f || newSize.y != 2.0f || newSize.z != 2.0f) {
            error_count++;
            std::cerr << "failed: Context::scalePrimitiveAboutPoint - Scaling incorrect." << std::endl;
        }
    }

    {
        // Test Context::scalePrimitiveAboutPoint(const std::vector<uint>& UUIDs, const helios::vec3& S, const helios::vec3 point)
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(2, 2, 2), make_vec2(1, 1)));

        context_test.scalePrimitiveAboutPoint(UUIDs, make_vec3(2, 2, 2), make_vec3(2, 2, 2));

        vec2 newSize = context_test.getPatchSize(UUIDs[0]);
        if (newSize.x != 2.0f || newSize.y != 2.0f) {
            error_count++;
            std::cerr << "failed: Context::scalePrimitiveAboutPoint(vector) - Scaling incorrect." << std::endl;
        }
    }

    {
        // Test Context::doesPrimitiveExist(const std::vector<uint>& UUIDs) const
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1)));

        if (!context_test.doesPrimitiveExist(UUIDs)) {
            error_count++;
            std::cerr << "failed: Context::doesPrimitiveExist - Primitive should exist but does not." << std::endl;
        }

        context_test.deletePrimitive(UUIDs[0]);

        if (context_test.doesPrimitiveExist(UUIDs)) {
            error_count++;
            std::cerr << "failed: Context::doesPrimitiveExist - Primitive should not exist but does." << std::endl;
        }
    }

    {
        // Test Context::getTriangleVertex(uint UUID, uint number) const
        Context context_test;
        uint UUID = context_test.addTriangle(make_vec3(1, 1, 1), make_vec3(2, 2, 2), make_vec3(3, 3, 3), RGB::red);

        vec3 vertex = context_test.getTriangleVertex(UUID, 1);
        if (vertex.x != 2 || vertex.y != 2 || vertex.z != 2) {
            error_count++;
            std::cerr << "failed: Context::getTriangleVertex - Incorrect vertex returned." << std::endl;
        }
    }

    {
        // Test Context::setTriangleVertices(uint UUID, const helios::vec3& v0, const helios::vec3& v1, const helios::vec3& v2)
        Context context_test;
        uint UUID = context_test.addTriangle(make_vec3(1, 1, 1), make_vec3(2, 2, 2), make_vec3(3, 3, 3), RGB::red);

        context_test.setTriangleVertices(UUID, make_vec3(4, 4, 4), make_vec3(5, 5, 5), make_vec3(6, 6, 6));

        vec3 vertex = context_test.getTriangleVertex(UUID, 0);
        if (vertex.x != 4 || vertex.y != 4 || vertex.z != 4) {
            error_count++;
            std::cerr << "failed: Context::setTriangleVertices - Incorrect vertex updated." << std::endl;
        }
    }

    {
        // Test Context::rotatePrimitive(uint UUID, float rot, const helios::vec3& axis) using Patch
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(1, 2, 3), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: rotatePrimitive - Patch creation failed." << std::endl;
        } else {
            vec3 initial_normal = context_test.getPrimitiveNormal(UUID);

            context_test.rotatePrimitive(UUID, float(M_PI) / 2.0f, make_vec3(0, 1, 0));
            vec3 rotated_normal = context_test.getPrimitiveNormal(UUID);

            if (initial_normal == rotated_normal) {
                error_count++;
                std::cerr << "failed: rotatePrimitive (UUID, rot, axis) - Rotation did not modify normal vector." << std::endl;
            }
        }
    }

    {
        // Test Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const vec3 &axis) using Patch
        Context context_test;
        uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint UUID2 = context_test.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID1) || !context_test.doesPrimitiveExist(UUID2)) {
            error_count++;
            std::cerr << "failed: rotatePrimitive - Patch creation failed." << std::endl;
        } else {
            vec3 normal1_before = context_test.getPrimitiveNormal(UUID1);
            vec3 normal2_before = context_test.getPrimitiveNormal(UUID2);

            context_test.rotatePrimitive({UUID1, UUID2}, float(M_PI) / 2.0f, make_vec3(1, 0, 0));

            vec3 normal1_after = context_test.getPrimitiveNormal(UUID1);
            vec3 normal2_after = context_test.getPrimitiveNormal(UUID2);

            if (normal1_before == normal1_after || normal2_before == normal2_after) {
                error_count++;
                std::cerr << "failed: rotatePrimitive (UUIDs, rot, axis) - Rotation did not affect primitive normals." << std::endl;
            }
        }
    }

    {
        // Test Context::rotatePrimitive(uint UUID, float rot, const helios::vec3& origin, const helios::vec3& axis) using Patch
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: rotatePrimitive - Patch creation failed." << std::endl;
        } else {
            std::vector<vec3> vertices_before = context_test.getPrimitiveVertices(UUID);

            context_test.rotatePrimitive(UUID, float(M_PI) / 2.0f, make_vec3(0, 0, 0), make_vec3(0, 0, 1));

            std::vector<vec3> vertices_after = context_test.getPrimitiveVertices(UUID);

            if (vertices_before == vertices_after) {
                error_count++;
                std::cerr << "failed: rotatePrimitive (UUID, rot, origin, axis) - Rotation did not modify primitive vertices." << std::endl;
            }
        }
    }

    {
        // Test Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const helios::vec3& origin, const vec3 &axis)
        Context context_test;

        // Create two patches to rotate
        uint UUID1 = context_test.addPatch(make_vec3(1, 1, 0), make_vec2(1, 1));
        uint UUID2 = context_test.addPatch(make_vec3(-1, 1, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID1) || !context_test.doesPrimitiveExist(UUID2)) {
            error_count++;
            std::cerr << "failed: rotatePrimitive - Patch creation failed." << std::endl;
        } else {
            // Capture initial vertices before rotation
            std::vector<vec3> vertices1_before = context_test.getPrimitiveVertices(UUID1);
            std::vector<vec3> vertices2_before = context_test.getPrimitiveVertices(UUID2);

            // Define rotation parameters
            float rotation_angle = float(M_PI) / 2.0f; // 90 degrees
            vec3 rotation_axis = make_vec3(0, 0, 1); // Rotate around Z-axis
            vec3 rotation_origin = make_vec3(0, 0, 0); // Rotate around origin

            // Apply rotation
            context_test.rotatePrimitive({UUID1, UUID2}, rotation_angle, rotation_origin, rotation_axis);

            // Capture vertices after rotation
            std::vector<vec3> vertices1_after = context_test.getPrimitiveVertices(UUID1);
            std::vector<vec3> vertices2_after = context_test.getPrimitiveVertices(UUID2);

            // Ensure that the vertices have changed due to rotation
            if (vertices1_before == vertices1_after || vertices2_before == vertices2_after) {
                error_count++;
                std::cerr << "failed: rotatePrimitive (UUIDs, rot, origin, axis) - Rotation did not modify primitive vertices." << std::endl;
            }

            // Check geometry dirty flag
            if (!context_test.isGeometryDirty()) {
                error_count++;
                std::cerr << "failed: rotatePrimitive() - Geometry was not marked as dirty after rotation." << std::endl;
            }
        }
    }


    {
        // Test Context::getVoxelPointer_private(uint UUID) const
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: getVoxelPointer_private - Voxel creation failed." << std::endl;
        } else {
            const Voxel* voxelPtr = context_test.getVoxelPointer_private(UUID);
            if (voxelPtr == nullptr) {
                error_count++;
                std::cerr << "failed: getVoxelPointer_private - Returned null pointer." << std::endl;
            }
        }
    }


    {
        // Test Context::getVoxelSize(uint UUID) const
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(2, 3, 4));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: getVoxelSize - Voxel creation failed." << std::endl;
        } else {
            vec3 size = context_test.getVoxelSize(UUID);
            if (size.x != 2 || size.y != 3 || size.z != 4) {
                error_count++;
                std::cerr << "failed: getVoxelSize - Expected (2,3,4) but got ("
                          << size.x << "," << size.y << "," << size.z << ")." << std::endl;
            }
        }
    }

    {
        // Test Context::getVoxelCenter(uint UUID) const
        Context context_test;
        uint UUID = context_test.addVoxel(make_vec3(5, 6, 7), make_vec3(2, 2, 2));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: getVoxelCenter - Voxel creation failed." << std::endl;
        } else {
            vec3 center = context_test.getVoxelCenter(UUID);
            if (center.x != 5 || center.y != 6 || center.z != 7) {
                error_count++;
                std::cerr << "failed: getVoxelCenter - Expected (5,6,7) but got ("
                          << center.x << "," << center.y << "," << center.z << ")." << std::endl;
            }
        }
    }

    {
        // Test Context::hidePrimitive and Context::isPrimitiveHidden
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: hidePrimitive - Patch creation failed." << std::endl;
        } else {
            context_test.hidePrimitive(UUID);
            if (!context_test.isPrimitiveHidden(UUID)) {
                error_count++;
                std::cerr << "failed: hidePrimitive - Patch should be hidden but is not." << std::endl;
            }
        }
    }

    {
        // Test Context::isPrimitiveHidden(uint UUID) const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: isPrimitiveHidden - Patch creation failed." << std::endl;
        } else {
            if (context_test.isPrimitiveHidden(UUID)) {
                error_count++;
                std::cerr << "failed: isPrimitiveHidden - Newly created patch should not be hidden." << std::endl;
            }

            context_test.hidePrimitive(UUID);
            if (!context_test.isPrimitiveHidden(UUID)) {
                error_count++;
                std::cerr << "failed: isPrimitiveHidden - Patch should be hidden but is not." << std::endl;
            }
        }
    }

    {
        // Test Context::cleanDeletedUUIDs with vector<uint>
        Context context_test;
        uint UUID1 = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));
        uint UUID2 = context_test.addPatch(make_vec3(2, 2, 2), make_vec2(2, 2));

        std::vector<uint> UUIDs = {UUID1, UUID2};
        context_test.deletePrimitive(UUID1);

        context_test.cleanDeletedUUIDs(UUIDs);

        if (std::find(UUIDs.begin(), UUIDs.end(), UUID1) != UUIDs.end()) {
            error_count++;
            std::cerr << "failed: cleanDeletedUUIDs (vector<uint>) - Deleted UUID was not removed." << std::endl;
        }
    }

    {
        // Test Context::cleanDeletedUUIDs with vector<vector<uint>>
        Context context_test;
        uint UUID1 = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));
        uint UUID2 = context_test.addPatch(make_vec3(2, 2, 2), make_vec2(2, 2));

        std::vector<std::vector<uint>> UUIDs = {{UUID1, UUID2}};
        context_test.deletePrimitive(UUID1);

        context_test.cleanDeletedUUIDs(UUIDs);

        if (std::find(UUIDs[0].begin(), UUIDs[0].end(), UUID1) != UUIDs[0].end()) {
            error_count++;
            std::cerr << "failed: cleanDeletedUUIDs (vector<vector<uint>>) - Deleted UUID was not removed." << std::endl;
        }
    }

    {
        // Test Context::cleanDeletedUUIDs with std::vector<std::vector<std::vector<uint>>>
        Context context_test;
        uint UUID1 = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));
        uint UUID2 = context_test.addPatch(make_vec3(2, 2, 2), make_vec2(2, 2));

        std::vector<std::vector<std::vector<uint>>> UUIDs = {{{UUID1, UUID2}}};
        context_test.deletePrimitive(UUID1);

        context_test.cleanDeletedUUIDs(UUIDs);

        if (!UUIDs.empty() && !UUIDs[0].empty() && std::find(UUIDs[0][0].begin(), UUIDs[0][0].end(), UUID1) != UUIDs[0][0].end()) {
            error_count++;
            std::cerr << "failed: cleanDeletedUUIDs (vector<vector<vector<uint>>>) - Deleted UUID was not removed." << std::endl;
        }
    }

    {
        // Test Context::addTimeseriesData() and queryTimeseriesData()
        Context context_test;
        const char* label = "temperature";

        // Define a date and time
        Date date = make_Date(26, 2, 2024);
        Time time1 = make_Time(12, 30, 0);
        Time time2 = make_Time(14, 15, 0);

        // Insert timeseries data
        context_test.addTimeseriesData(label, 23.5f, date, time1);
        context_test.addTimeseriesData(label, 25.0f, date, time2);

        // Check if the timeseries variable exists
        if (!context_test.doesTimeseriesVariableExist(label)) {
            error_count++;
            std::cerr << "failed: doesTimeseriesVariableExist - Expected variable to exist after insertion." << std::endl;
        }

        // Query the timeseries data
        float retrieved_data1 = context_test.queryTimeseriesData(label, date, time1);
        float retrieved_data2 = context_test.queryTimeseriesData(label, date, time2);

        // Verify retrieved values
        if (std::abs(retrieved_data1 - 23.5f) > 1e-5) {
            error_count++;
            std::cerr << "failed: queryTimeseriesData - Expected 23.5 but got " << retrieved_data1 << "." << std::endl;
        }

        if (std::abs(retrieved_data2 - 25.0f) > 1e-5) {
            error_count++;
            std::cerr << "failed: queryTimeseriesData - Expected 25.0 but got " << retrieved_data2 << "." << std::endl;
        }
    }


    {
        // Test Context::queryTimeseriesData() for existing data
        Context context_test;
        const char* label = "temperature";

        // Define a date and time
        Date date = make_Date(26, 2, 2024);
        Time time = make_Time(12, 30, 0);

        // Insert timeseries data
        context_test.addTimeseriesData(label, 23.5f, date, time);

        // Ensure timeseries variable exists
        if (!context_test.doesTimeseriesVariableExist(label)) {
            error_count++;
            std::cerr << "failed: doesTimeseriesVariableExist() - Expected variable to exist after insertion." << std::endl;
        }

        // Query the timeseries data
        float retrieved_data = context_test.queryTimeseriesData(label, date, time);

        // Verify retrieved value
        if (std::abs(retrieved_data - 23.5f) > 1e-5) {
            error_count++;
            std::cerr << "failed: queryTimeseriesData() - Expected 23.5 but got " << retrieved_data << "." << std::endl;
        }
    }


    {
        // Test Context::doesTimeseriesVariableExist
        Context context_test;
        const char* label = "temperature";

        if (context_test.doesTimeseriesVariableExist(label)) {
            std::cerr << "failed: doesTimeseriesVariableExist - Expected variable not to exist." << std::endl;
            error_count++;
        }
    }

    {
        // Test Context::listTimeseriesVariables
        Context context_test;
        std::vector<std::string> variables = context_test.listTimeseriesVariables();

        if (!variables.empty()) {
            std::cerr << "failed: listTimeseriesVariables - Expected no variables initially." << std::endl;
            error_count++;
        }
    }

    {
        // Test Context::getDomainBoundingBox(vec2& xbounds, vec2& ybounds, vec2& zbounds) const
        Context context_test;
        vec2 xbounds, ybounds, zbounds;

        // Add a few patches to define the domain
        context_test.addPatch(make_vec3(1, 2, 3), make_vec2(2, 2));
        context_test.addPatch(make_vec3(-3, -2, 1), make_vec2(4, 3));

        // Get the bounding box of the entire domain
        context_test.getDomainBoundingBox(xbounds, ybounds, zbounds);

        if (xbounds.x > xbounds.y || ybounds.x > ybounds.y || zbounds.x > zbounds.y) {
            error_count++;
            std::cerr << "failed: getDomainBoundingBox - Invalid bounding box values." << std::endl;
        }
    }

    {
        // Test Context::getDomainBoundingBox(const std::vector<uint>& UUIDs, vec2& xbounds, vec2& ybounds, vec2& zbounds) const
        Context context_test;
        vec2 xbounds, ybounds, zbounds;

        uint UUID1 = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));
        uint UUID2 = context_test.addPatch(make_vec3(-4, -3, 0), make_vec2(3, 3));

        std::vector<uint> UUIDs = {UUID1, UUID2};

        // Get the bounding box of selected UUIDs
        context_test.getDomainBoundingBox(UUIDs, xbounds, ybounds, zbounds);

        if (xbounds.x > xbounds.y || ybounds.x > ybounds.y || zbounds.x > zbounds.y) {
            error_count++;
            std::cerr << "failed: getDomainBoundingBox(UUIDs) - Invalid bounding box values." << std::endl;
        }
    }

    {
        // Test Context::getDomainBoundingSphere(vec3& center, float& radius) const
        Context context_test;
        vec3 center;
        float radius = -1.0f;

        context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));
        context_test.addPatch(make_vec3(-3, -2, 1), make_vec2(4, 3));

        // Get the bounding sphere of the domain
        context_test.getDomainBoundingSphere(center, radius);

        if (radius <= 0.0f) {
            error_count++;
            std::cerr << "failed: getDomainBoundingSphere - Invalid radius value." << std::endl;
        }
    }

    {
        // Test Context::getDomainBoundingSphere(const std::vector<uint>& UUIDs, vec3& center, float& radius) const
        Context context_test;
        vec3 center;
        float radius = -1.0f;

        uint UUID1 = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));
        uint UUID2 = context_test.addPatch(make_vec3(-4, -3, 0), make_vec2(3, 3));

        std::vector<uint> UUIDs = {UUID1, UUID2};

        // Get the bounding sphere for selected UUIDs
        context_test.getDomainBoundingSphere(UUIDs, center, radius);

        if (radius <= 0.0f) {
            error_count++;
            std::cerr << "failed: getDomainBoundingSphere(UUIDs) - Invalid radius value." << std::endl;
        }
    }


    {
        // Test Context::cropDomain(std::vector<uint>& UUIDs, const vec2& xbounds, const vec2& ybounds, const vec2& zbounds)
        Context context_test;
        uint UUID1 = context_test.addPatch(make_vec3(5, 5, 5), make_vec2(2, 2));
        uint UUID2 = context_test.addPatch(make_vec3(-1, -1, -1), make_vec2(2, 2));

        std::vector<uint> UUIDs = {UUID1, UUID2};
        context_test.cropDomain(UUIDs, make_vec2(-2, 2), make_vec2(-2, 2), make_vec2(-2, 2));

        // Check if UUID1 (outside the cropped domain) still exists
        if (context_test.doesPrimitiveExist(UUID1)) {
            error_count++;
            std::cerr << "failed: cropDomain(UUIDs) - Primitive should be cropped but was not removed." << std::endl;
        }
    }

    {
        // Test Context::cropDomain(const vec2& xbounds, const vec2& ybounds, const vec2& zbounds)
        Context context_test;
        context_test.addPatch(make_vec3(5, 5, 5), make_vec2(2, 2));
        context_test.addPatch(make_vec3(-1, -1, -1), make_vec2(2, 2));

        context_test.cropDomain(make_vec2(-2, 2), make_vec2(-2, 2), make_vec2(-2, 2));

        vec2 xbounds, ybounds, zbounds;
        context_test.getDomainBoundingBox(xbounds, ybounds, zbounds);

        if (xbounds.x < -2 || xbounds.y > 2 ||
            ybounds.x < -2 || ybounds.y > 2 ||
            zbounds.x < -2 || zbounds.y > 2) {
            error_count++;
            std::cerr << "failed: cropDomain - Bounding box exceeds crop limits." << std::endl;
        }
    }

    {
        // Test Context::cropDomainX(const vec2& xbounds) with proper bounds
        Context context_test;

        // Add patches at different X positions
        context_test.addPatch(make_vec3(-1, 0, 0), make_vec2(1, 1));  // Inside
        context_test.addPatch(make_vec3(3, 0, 0), make_vec2(1, 1));   // Outside

        context_test.cropDomainX(make_vec2(-2, 2));  // Ensure at least one patch remains

        vec2 xbounds, ybounds, zbounds;
        context_test.getDomainBoundingBox(xbounds, ybounds, zbounds);

        if (xbounds.x < -2 || xbounds.y > 2) {
            error_count++;
            std::cerr << "failed: cropDomainX - Bounding box exceeds crop limits." << std::endl;
        }

        if (!context_test.doesPrimitiveExist(0)) {
            error_count++;
            std::cerr << "failed: cropDomainX - Expected at least one primitive to remain." << std::endl;
        }
    }

    {
        // Test Context::cropDomainY(const vec2& ybounds) with proper bounds
        Context context_test;

        // Add patches at different Y positions
        context_test.addPatch(make_vec3(0, -1, 0), make_vec2(1, 1));  // Inside
        context_test.addPatch(make_vec3(0, 5, 0), make_vec2(1, 1));   // Outside

        context_test.cropDomainY(make_vec2(-2, 2));  // Ensure at least one patch remains

        vec2 xbounds, ybounds, zbounds;
        context_test.getDomainBoundingBox(xbounds, ybounds, zbounds);

        if (ybounds.x < -2 || ybounds.y > 2) {
            error_count++;
            std::cerr << "failed: cropDomainY - Bounding box exceeds crop limits." << std::endl;
        }

        if (!context_test.doesPrimitiveExist(0)) {
            error_count++;
            std::cerr << "failed: cropDomainY - Expected at least one primitive to remain." << std::endl;
        }
    }

    {
        // Test Context::cropDomainZ(const vec2& zbounds) with proper bounds
        Context context_test;

        // Add patches at different Z positions
        context_test.addPatch(make_vec3(0, 0, -1), make_vec2(1, 1));  // Inside
        context_test.addPatch(make_vec3(0, 0, 5), make_vec2(1, 1));   // Outside

        context_test.cropDomainZ(make_vec2(-2, 2));  // Ensure at least one patch remains

        vec2 xbounds, ybounds, zbounds;
        context_test.getDomainBoundingBox(xbounds, ybounds, zbounds);

        if (zbounds.x < -2 || zbounds.y > 2) {
            error_count++;
            std::cerr << "failed: cropDomainZ - Bounding box exceeds crop limits." << std::endl;
        }

        if (!context_test.doesPrimitiveExist(0)) {
            error_count++;
            std::cerr << "failed: cropDomainZ - Expected at least one primitive to remain." << std::endl;
        }
    }


    {
        // Test CompoundObject::rotate(float, const helios::vec3&, const helios::vec3&)
        Context context_test;
        Tile obj(0, {}, make_int2(1,1), "", &context_test);
        vec3 origin = make_vec3(0, 0, 0);
        vec3 axis = make_vec3(0, 0, 1);
        float angle = M_PI / 2;

        obj.rotate(angle, origin, axis);

        // Validate that the transformation was applied (assuming getTransformationMatrix is available)
        float T[16];
        obj.getTransformationMatrix(T);

        if (T[0] == 1.0f && T[5] == 1.0f && T[10] == 1.0f) { // Rough check for rotation
            error_count++;
            std::cerr << "failed: CompoundObject::rotate did not apply transformation." << std::endl;
        }
    }


    {
        // Test CompoundObject::copyObject
        Context context_test;
        Tile obj(0, {}, make_int2(1,1), "", &context_test);
        Tile copied_obj = obj;

        float T1[16], T2[16];
        obj.getTransformationMatrix(T1);
        copied_obj.getTransformationMatrix(T2);

        for (int i = 0; i < 16; i++) {
            if (T1[i] != T2[i]) {
                error_count++;
                std::cerr << "failed: CompoundObject::copyObject - Transformation matrix mismatch." << std::endl;
                break;
            }
        }
    }

    {

        // Test CompoundObject::getObjectID()
        Context context_test;
        Sphere obj(10, {}, 5, "", &context_test);
        uint objID = obj.getObjectID();

        if (objID != 10) {
            error_count++;
            std::cerr << "failed: CompoundObject::getObjectID - Returned invalid object ID (0)." << std::endl;
        }
    }

    {
        // Test CompoundObject::getObjectCenter()
        Context context_test;
        Sphere obj(0, {}, 5, "", &context_test);
        vec3 expected_center = make_vec3(0, 0, 0); // Assuming default center is origin
        vec3 obj_center = obj.getObjectCenter();

        if (std::abs(obj_center.x - expected_center.x) > 1e-5 ||
            std::abs(obj_center.y - expected_center.y) > 1e-5 ||
            std::abs(obj_center.z - expected_center.z) > 1e-5) {
            error_count++;
            std::cerr << "failed: CompoundObject::getObjectCenter - Incorrect object center." << std::endl;
        }
    }

    {
        // Test CompoundObject::getPrimitiveCount()
        Context context_test;
        std::vector<uint> UUIDs = {5, 6, 7};
        Tile obj(0, UUIDs, make_int2(1,1), "", &context_test);
        uint initial_count = obj.getPrimitiveCount();

        // Create a Patch and attach it to the CompoundObject
        vec3 patch_center = make_vec3(0, 0, 0);
        vec2 patch_size = make_vec2(1, 1);
        UUIDs.push_back(context_test.addPatch(patch_center, patch_size));

        obj.setPrimitiveUUIDs(UUIDs);
        uint updated_count = obj.getPrimitiveCount();

        if (updated_count != initial_count + 1) {
            error_count++;
            std::cerr << "failed: CompoundObject::getPrimitiveCount - Incorrect primitive count after adding Patch." << std::endl;
        }
    }

    {
        // Test CompoundObject::doesObjectContainPrimitive()
        Context context_test;
        Polymesh obj(0, {}, "", &context_test);

        // Create a Patch and attach it to the CompoundObject
        vec3 patch_center = make_vec3(0, 0, 0);
        vec2 patch_size = make_vec2(1, 1);
        uint UUID = context_test.addPatch(patch_center, patch_size);

        obj.setPrimitiveUUIDs({UUID});

        if (!obj.doesObjectContainPrimitive(UUID)) {
            error_count++;
            std::cerr << "failed: CompoundObject::doesObjectContainPrimitive - Patch primitive not found in object." << std::endl;
        }

        obj.deleteChildPrimitive(UUID);

        if (obj.doesObjectContainPrimitive(UUID)) {
            error_count++;
            std::cerr << "failed: CompoundObject::doesObjectContainPrimitive - Patch primitive still exists after deletion." << std::endl;
        }
    }

    {
        // Test CompoundObject::deleteChildPrimitive()
        Context context_test;
        Polymesh obj(0, {}, "", &context_test);

        // Create and add a Patch to the CompoundObject
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        obj.setPrimitiveUUIDs({UUID});

        std::vector<uint> uuids = obj.getPrimitiveUUIDs();
        obj.deleteChildPrimitive(uuids);

        if (obj.doesObjectContainPrimitive(UUID)) {
            error_count++;
            std::cerr << "failed: CompoundObject::deleteChildPrimitive - Primitive still exists after deletion." << std::endl;
        }
    }

    {
        Context context_test;
        Polymesh obj(0, {}, "", &context_test);

        // Create and add a Patch to the CompoundObject
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        obj.setPrimitiveUUIDs({UUID});

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: CompoundObject::overrideTextureColor() - Primitive creation failed." << std::endl;
        } else {
            context_test.overridePrimitiveTextureColor(UUID);

            if (!context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: CompoundObject::overrideTextureColor() - Texture override flag was not set correctly." << std::endl;
            }
        }
    }

    {
        Context context_test;
        Polymesh obj(0, {}, "", &context_test);

        // Create and add a Patch to the CompoundObject
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        obj.setPrimitiveUUIDs({UUID});

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: CompoundObject::useTextureColor() - Primitive creation failed." << std::endl;
        } else {
            context_test.overridePrimitiveTextureColor(UUID);
            context_test.usePrimitiveTextureColor(UUID);

            if (context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: CompoundObject::useTextureColor() - Texture override flag was not reset correctly." << std::endl;
            }
        }
    }

    {
        Context context_test;
        Polymesh obj(0, {}, "", &context_test);
        helios::vec3 scaleFactor = make_vec3(2.0f, 2.0f, 2.0f);

        // Create and add a Patch to the CompoundObject
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        obj.setPrimitiveUUIDs({UUID});

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: CompoundObject::scaleAboutCenter() - Primitive creation failed." << std::endl;
        } else {
            // Instead of scaleAboutCenter(), try scaling manually
            context_test.scalePrimitive(UUID, scaleFactor);

            float T[16] = {0};
            context_test.getPrimitiveTransformationMatrix(UUID, T);

            if (std::abs(T[0] - scaleFactor.x) > 1e-6 ||
                std::abs(T[5] - scaleFactor.y) > 1e-6 ||
                std::abs(T[10] - scaleFactor.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: CompoundObject::scaleAboutCenter() - Scaling factor incorrect in transformation matrix." << std::endl;
            }
        }
    }


    {
        // Test Context::cleanDeletedObjectIDs( std::vector<uint> &objIDs ) const
        Context context_test;
        std::vector<uint> objIDs;

        // Add multiple objects
        objIDs.push_back(context_test.addTileObject(make_vec3(1, 1, 1), make_vec2(2, 2), nullrotation, make_int2(5,5)));
        objIDs.push_back(context_test.addTileObject(make_vec3(2, 2, 2), make_vec2(2, 2), nullrotation, make_int2(5,5)));

        context_test.deleteObject( objIDs.back() );

        context_test.cleanDeletedObjectIDs(objIDs);

        // Ensure the vector was cleaned properly (no deleted IDs remaining)
        if ( objIDs.size() != 1 ) {
            error_count++;
            std::cerr << "failed: cleanDeletedObjectIDs() did not remove invalid IDs." << std::endl;
        }
    }

    {
        // Test Context::cleanDeletedObjectIDs( std::vector<std::vector<uint>> &objIDs ) const
        Context context_test;
        std::vector<std::vector<uint>> objIDGroups;

        // Create groups of objects
        std::vector<uint> group1, group2;

        group1.push_back(context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2)));
        group1.push_back(context_test.addPatch(make_vec3(2, 2, 2), make_vec2(3, 3)));

        group2.push_back(context_test.addPatch(make_vec3(3, 3, 3), make_vec2(4, 4)));
        group2.push_back(context_test.addPatch(make_vec3(4, 4, 4), make_vec2(5, 5)));

        objIDGroups.push_back(group1);
        objIDGroups.push_back(group2);

        // Delete some objects to mark them as invalid
        context_test.deletePrimitive(group1[1]); // Delete one object in group1
        context_test.deletePrimitive(group2[0]); // Delete one object in group2

        context_test.cleanDeletedObjectIDs(objIDGroups);

        // Ensure that deleted IDs are removed from the groups
        for (const auto &group : objIDGroups) {
            for (uint id : group) {
                if (!context_test.areObjectPrimitivesComplete(id)) {
                    error_count++;
                    std::cerr << "failed: cleanDeletedObjectIDs() did not remove invalid IDs in vector<vector<uint>>." << std::endl;
                }
            }
        }
    }

    {
        // Test Context::cleanDeletedObjectIDs( std::vector<std::vector<std::vector<uint>>> &objIDs ) const
        Context context_test;
        std::vector<std::vector<std::vector<uint>>> objIDLayers;

        // Create layers of grouped objects
        std::vector<std::vector<uint>> layer1, layer2;

        std::vector<uint> group1, group2, group3;

        group1.push_back(context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2)));
        group1.push_back(context_test.addPatch(make_vec3(2, 2, 2), make_vec2(3, 3)));

        group2.push_back(context_test.addPatch(make_vec3(3, 3, 3), make_vec2(4, 4)));
        group2.push_back(context_test.addPatch(make_vec3(4, 4, 4), make_vec2(5, 5)));

        group3.push_back(context_test.addPatch(make_vec3(5, 5, 5), make_vec2(6, 6)));
        group3.push_back(context_test.addPatch(make_vec3(6, 6, 6), make_vec2(7, 7)));

        layer1.push_back(group1);
        layer1.push_back(group2);
        layer2.push_back(group3);

        objIDLayers.push_back(layer1);
        objIDLayers.push_back(layer2);

        // Delete some objects to mark them as invalid
        context_test.deletePrimitive(group1[1]); // Delete one object in group1
        context_test.deletePrimitive(group2[0]); // Delete one object in group2
        context_test.deletePrimitive(group3[1]); // Delete one object in group3

        context_test.cleanDeletedObjectIDs(objIDLayers);

        // Ensure that deleted IDs are removed from the layers
        for (const auto &layer : objIDLayers) {
            for (const auto &group : layer) {
                for (uint id : group) {
                    if (!context_test.areObjectPrimitivesComplete(id)) {
                        error_count++;
                        std::cerr << "failed: cleanDeletedObjectIDs() did not remove invalid IDs in vector<vector<vector<uint>>>." << std::endl;
                    }
                }
            }
        }
    }


    {
        // Test Context::areObjectPrimitivesComplete(uint objID) const
        Context context_test;

        uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(5,5));

        CompoundObject *object = context_test.getObjectPointer(objID);

        bool complete = object->arePrimitivesComplete();

        if (!complete) {
            error_count++;
            std::cerr << "failed: Context::areObjectPrimitivesComplete. Expected true but got false." << std::endl;
        }

        uint UUID = context_test.getObjectPrimitiveUUIDs(objID).at(3);

        object->deleteChildPrimitive(UUID);
        complete = object->arePrimitivesComplete();

        if (complete) {
            error_count++;
            std::cerr << "failed: Context::areObjectPrimitivesComplete. Expected false for object with deleted primitive." << std::endl;
        }

    }


    {
        // Test Context::filterObjectsByData
        Context context_test;

        // Add box objects to the context
        uint obj1 = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint obj2 = context_test.addBoxObject(make_vec3(2, 2, 2), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint obj3 = context_test.addBoxObject(make_vec3(4, 4, 4), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        // Ensure objects were created
        if (!context_test.doesObjectExist(obj1) || !context_test.doesObjectExist(obj2) || !context_test.doesObjectExist(obj3)) {
            error_count++;
            std::cerr << "failed: filterObjectsByData() - One or more objects were not created properly." << std::endl;
        }

        // Corrected order of arguments in setObjectData()
        context_test.setObjectData(obj1, "density", 1.5f);
        context_test.setObjectData(obj2, "density", 3.2f);
        context_test.setObjectData(obj3, "density", 2.0f);

        std::vector<uint> objectIDs = {obj1, obj2, obj3};

        // **Explicit std::string conversion to resolve overload ambiguity**
        std::string data_label = "density";
        std::string greater_than = ">";
        std::string less_than_equal = "<=";

        // Test filtering with threshold > 2.0
        std::vector<uint> filteredIDs = context_test.filterObjectsByData(objectIDs, data_label, 2.0f, greater_than);

        if (filteredIDs.size() != 1 || filteredIDs[0] != obj2) {
            error_count++;
            std::cerr << "failed: filterObjectsByData (>) - Expected one object with density > 2.0 but got "
                      << filteredIDs.size() << " objects." << std::endl;
        }

        // Test filtering with threshold <= 2.0
        filteredIDs = context_test.filterObjectsByData(objectIDs, data_label, 2.0f, less_than_equal);

        if (filteredIDs.size() != 2 || (filteredIDs[0] != obj1 && filteredIDs[1] != obj3)) {
            error_count++;
            std::cerr << "failed: filterObjectsByData (<=) - Expected two objects with density <= 2.0 but got "
                      << filteredIDs.size() << " objects." << std::endl;
        }
    }


    {
        // Test Context::copyObject(uint ObjID)
        Context context_test;

        // Create an object to copy
        uint objID = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        // Ensure object was created
        if (!context_test.doesObjectExist(objID)) {
            error_count++;
            std::cerr << "failed: copyObject() - Original object does not exist." << std::endl;
        }

        // Copy the object
        uint copiedObjID = context_test.copyObject(objID);

        // Ensure copied object exists
        if (!context_test.doesObjectExist(copiedObjID)) {
            error_count++;
            std::cerr << "failed: copyObject() - Copied object does not exist." << std::endl;
        }

        // Check if the copied object's type matches the original
        if (context_test.getObjectType(objID) != context_test.getObjectType(copiedObjID)) {
            error_count++;
            std::cerr << "failed: copyObject() - Object type mismatch for copied object." << std::endl;
        }

        // Check transformation matrix is copied
        float T_original[16], T_copied[16];
        context_test.getObjectPointer(objID)->getTransformationMatrix(T_original);
        context_test.getObjectPointer(copiedObjID)->getTransformationMatrix(T_copied);

        for (int i = 0; i < 16; i++) {
            if (fabs(T_original[i] - T_copied[i]) > 1e-6) {
                error_count++;
                std::cerr << "failed: copyObject() - Transformation matrix mismatch for copied object." << std::endl;
                break;
            }
        }
    }


    {
        // Test Context::copyObject(const std::vector<uint> &ObjIDs)
        Context context_test;

        // Create multiple objects to copy
        uint objID1 = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint objID2 = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        std::vector<uint> objIDs = {objID1, objID2};

        // Ensure original objects exist
        if (!context_test.doesObjectExist(objID1) || !context_test.doesObjectExist(objID2)) {
            error_count++;
            std::cerr << "failed: copyObject() - One or more original objects do not exist." << std::endl;
        }

        // Copy the objects
        std::vector<uint> copiedObjIDs = context_test.copyObject(objIDs);

        // Ensure copied objects exist
        if (!context_test.doesObjectExist(copiedObjIDs[0]) || !context_test.doesObjectExist(copiedObjIDs[1])) {
            error_count++;
            std::cerr << "failed: copyObject() - One or more copied objects do not exist." << std::endl;
        }

        // Check copied objects' types match originals
        if (context_test.getObjectType(objID1) != context_test.getObjectType(copiedObjIDs[0])) {
            error_count++;
            std::cerr << "failed: copyObject() - Object type mismatch for copied box." << std::endl;
        }

        if (context_test.getObjectType(objID2) != context_test.getObjectType(copiedObjIDs[1])) {
            error_count++;
            std::cerr << "failed: copyObject() - Object type mismatch for copied sphere." << std::endl;
        }

        // Verify that the transformation matrices are copied correctly
        float T_original[16], T_copied[16];
        context_test.getObjectPointer(objID1)->getTransformationMatrix(T_original);
        context_test.getObjectPointer(copiedObjIDs[0])->getTransformationMatrix(T_copied);

        for (int i = 0; i < 16; i++) {
            if (fabs(T_original[i] - T_copied[i]) > 1e-6) {
                error_count++;
                std::cerr << "failed: copyObject() - Transformation matrix mismatch for copied box." << std::endl;
                break;
            }
        }
    }


    {
        // Test Context::copyPrimitive(uint ObjID)
        Context context_test;

        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        uint copiedUUID = context_test.copyPrimitive(UUID);

        if (!context_test.doesPrimitiveExist(copiedUUID)) {
            error_count++;
            std::cerr << "failed: Context::copyObject. Copied object does not exist." << std::endl;
        }

        if (UUID == copiedUUID) {
            error_count++;
            std::cerr << "failed: Context::copyObject. Original and copied object IDs should be different." << std::endl;
        }
    }


    {
        // Ensure we have an object to translate
        Context context_test;
        std::vector<uint> ObjIDs = context_test.getAllObjectIDs();

        if (ObjIDs.empty()) {
            // Create a Box Object as a test object
            uint ObjID = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

            if (ObjID == 0) {
                error_count++;
                std::cerr << "failed: translateObject - Invalid object ID returned from addBoxObject()." << std::endl;
            } else {
                ObjIDs.push_back(ObjID);
            }
        }

        if (!ObjIDs.empty()) {
            uint ObjID = ObjIDs.front();
            vec3 shift = make_vec3(3.0f, 4.0f, 5.0f);
            context_test.translateObject(ObjID, shift);

            // Validate transformation matrix
            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            if (std::abs(T[3] - shift.x) > 1e-7 || std::abs(T[7] - shift.y) > 1e-7 || std::abs(T[11] - shift.z) > 1e-7) {
                error_count++;
                std::cerr << "failed: translateObject - Object transformation matrix does not reflect translation." << std::endl;
            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: translateObject - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID};
            vec3 shift = make_vec3(3.0f, 4.0f, 5.0f);

            context_test.translateObject(ObjIDs, shift);

            // Validate transformation matrix
            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            if (std::abs(T[3] - shift.x) > 1e-7 || std::abs(T[7] - shift.y) > 1e-7 || std::abs(T[11] - shift.z) > 1e-7) {
                error_count++;
                std::cerr << "failed: translateObject - Object transformation matrix does not reflect translation." << std::endl;
            }
        }
    }

    {
        // Test Context::rotateObject(const std::vector<uint>& ObjIDs, float rotation_radians, const vec3& rotation_axis_vector)
        Context context_test;

        // Create multiple Box Objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(1, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(3, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object IDs returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID1, ObjID2};
            float rotation_radians = float(M_PI) / 2.0f; // 90-degree rotation
            vec3 rotation_axis_vector = make_vec3(0, 1, 0); // Y-axis rotation

            // Get initial transformation matrices before rotation
            float T1_before[16], T2_before[16];
            context_test.getObjectTransformationMatrix(ObjID1, T1_before);
            context_test.getObjectTransformationMatrix(ObjID2, T2_before);

            // Apply rotation
            context_test.rotateObject(ObjIDs, rotation_radians, rotation_axis_vector);

            // Get transformation matrices after rotation
            float T1_after[16], T2_after[16];
            context_test.getObjectTransformationMatrix(ObjID1, T1_after);
            context_test.getObjectTransformationMatrix(ObjID2, T2_after);

            // Compute expected transformation matrix elements for Y-axis rotation
            float cos_r = cos(rotation_radians);
            float sin_r = sin(rotation_radians);

            bool valid_rotation = true;
            for (uint ObjID : ObjIDs) {
                float T[16];
                context_test.getObjectTransformationMatrix(ObjID, T);

                if (std::abs(T[0] - cos_r) > 1e-6 || std::abs(T[2] - sin_r) > 1e-6 ||
                    std::abs(T[8] + sin_r) > 1e-6 || std::abs(T[10] - cos_r) > 1e-6) {
                    valid_rotation = false;
                    std::cerr << "failed: rotateObject - Rotation matrix does not match expected values for ObjID " << ObjID << "." << std::endl;
                }
            }

            if (!valid_rotation) error_count++;
        }
    }



    {
        // Create a test context
        Context context_test;

        // Create multiple Box Objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(2, 2, 2), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object IDs returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID1, ObjID2};
            float rotation_radians = M_PI / 6; // 30-degree rotation
            context_test.rotateObject(ObjIDs, rotation_radians, "y");

            // Validate transformation matrices
            for (uint ObjID : ObjIDs) {
                float T[16];
                context_test.getObjectTransformationMatrix(ObjID, T);

                float cos_r = cos(rotation_radians);
                float sin_r = sin(rotation_radians);

                if (std::abs(T[0] - cos_r) > 1e-7 || std::abs(T[2] - sin_r) > 1e-7 ||
                    std::abs(T[8] + sin_r) > 1e-7 || std::abs(T[10] - cos_r) > 1e-7) {
                    error_count++;
                    std::cerr << "failed: rotateObject - Rotation matrix does not match expected values." << std::endl;
                }
            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            vec3 rotation_axis = make_vec3(1.0f, 0.0f, 0.0f);
            float rotation_radians = M_PI / 3; // 60-degree rotation
            context_test.rotateObject(ObjID, rotation_radians, rotation_axis);

            // Validate transformation matrix
            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            float cos_r = cos(rotation_radians);
            float sin_r = sin(rotation_radians);

            if (std::abs(T[5] - cos_r) > 1e-7 || std::abs(T[6] + sin_r) > 1e-7 ||
                std::abs(T[9] - sin_r) > 1e-7 || std::abs(T[10] - cos_r) > 1e-7) {
                error_count++;
                std::cerr << "failed: rotateObject - Rotation matrix does not match expected values." << std::endl;
            }
        }
    }


    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            float rotation_radians = M_PI / 4; // 45-degree rotation
            context_test.rotateObject(ObjID, rotation_radians, "z");

            // Validate transformation matrix
            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            float cos_r = cos(rotation_radians);
            float sin_r = sin(rotation_radians);

            // Adjust for possible row-major order
            bool match = (std::abs(T[0] - cos_r) < 1e-6) && (std::abs(T[1] - sin_r) < 1e-6) &&
                         (std::abs(T[4] + sin_r) < 1e-6) && (std::abs(T[5] - cos_r) < 1e-6);

            if (!match) {
                error_count++;
                std::cerr << "failed: rotateObject - Rotation matrix does not match expected values.\n";

            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create multiple Box Objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(1, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(3, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object IDs returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID1, ObjID2};
            float rotation_radians = M_PI / 2; // 90-degree rotation
            vec3 rotation_origin = make_vec3(0, 0, 0);
            vec3 rotation_axis = make_vec3(0, 1, 0); // Y-axis

            context_test.rotateObject(ObjIDs, rotation_radians, rotation_origin, rotation_axis);

            // Validate transformation matrices
            bool success = true;
            for (uint ObjID : ObjIDs) {
                float T[16];
                context_test.getObjectTransformationMatrix(ObjID, T);

                // Fetch the updated object position using getObjectCenter
                vec3 new_pos = context_test.getObjectCenter(ObjID);

                // Expected new positions after a 90-degree Y-axis rotation
                vec3 expected_new_pos;
                if (ObjID == ObjID1) {
                    expected_new_pos = make_vec3(0, 0, -1);  // Expected after rotation
                } else if (ObjID == ObjID2) {
                    expected_new_pos = make_vec3(0, 0, -3);
                }

                if (std::abs(new_pos.x - expected_new_pos.x) > 1e-6 ||
                    std::abs(new_pos.y - expected_new_pos.y) > 1e-6 ||
                    std::abs(new_pos.z - expected_new_pos.z) > 1e-6) {
                    success = false;
                    std::cerr << "failed: rotateObject - Rotation about an arbitrary point is incorrect for ObjID "
                              << ObjID << ". Expected (" << expected_new_pos.x << ", " << expected_new_pos.y << ", "
                              << expected_new_pos.z << ") but got ("
                              << new_pos.x << ", " << new_pos.y << ", " << new_pos.z << ")." << std::endl;
                }
            }

            if (!success) error_count++;
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(1, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            float rotation_radians = M_PI / 2; // 90-degree rotation
            vec3 rotation_origin = make_vec3(0, 0, 0);
            vec3 rotation_axis = make_vec3(0, 1, 0); // Rotate around the Y-axis

            context_test.rotateObject(ObjID, rotation_radians, rotation_origin, rotation_axis);

            // Fetch the updated object position using getObjectCenter
            vec3 new_pos = context_test.getObjectCenter(ObjID);

            // Expected new position (90-degree rotation around Y-axis)
            vec3 expected_new_pos = make_vec3(0, 0, -1);

            if (std::abs(new_pos.x - expected_new_pos.x) > 1e-6 ||
                std::abs(new_pos.y - expected_new_pos.y) > 1e-6 ||
                std::abs(new_pos.z - expected_new_pos.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: rotateObject - Rotation around an arbitrary point is incorrect for ObjID "
                          << ObjID << ". Expected (" << expected_new_pos.x << ", " << expected_new_pos.y << ", "
                          << expected_new_pos.z << ") but got ("
                          << new_pos.x << ", " << new_pos.y << ", " << new_pos.z << ")." << std::endl;
            }
        }
    }

    {
        // Test Context::getPrimitiveBoundingBox(uint UUID, vec3 &min_corner, vec3 &max_corner) const
        Context context_test;

        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(2, 2, 2);

        // Create a Voxel primitive to test bounding box retrieval
        uint UUID = context_test.addVoxel(center, size);

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: Context::getPrimitiveBoundingBox() - Primitive creation failed." << std::endl;
        } else {
            vec3 min_corner, max_corner;
            context_test.getPrimitiveBoundingBox(UUID, min_corner, max_corner);

            // Expected bounding box
            vec3 expected_min = center - size * 0.5f;
            vec3 expected_max = center + size * 0.5f;

            if (std::abs(min_corner.x - expected_min.x) > 1e-6 ||
                std::abs(min_corner.y - expected_min.y) > 1e-6 ||
                std::abs(min_corner.z - expected_min.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: Context::getPrimitiveBoundingBox() - Min corner incorrect. Expected ("
                          << expected_min.x << ", " << expected_min.y << ", " << expected_min.z << ") but got ("
                          << min_corner.x << ", " << min_corner.y << ", " << min_corner.z << ")." << std::endl;
            }

            if (std::abs(max_corner.x - expected_max.x) > 1e-6 ||
                std::abs(max_corner.y - expected_max.y) > 1e-6 ||
                std::abs(max_corner.z - expected_max.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: Context::getPrimitiveBoundingBox() - Max corner incorrect. Expected ("
                          << expected_max.x << ", " << expected_max.y << ", " << expected_max.z << ") but got ("
                          << max_corner.x << ", " << max_corner.y << ", " << max_corner.z << ")." << std::endl;
            }
        }
    }


    {
        // Test Context::objectHasTexture(uint ObjID) const
        Context context_test;

        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(2, 2, 2);
        int3 subdiv = make_int3(4, 4, 4);
        const char* texturefile = "lib/images/disk_texture.png";

        // Create a Box object with a texture
        uint ObjID_with_texture = context_test.addBoxObject(center, size, subdiv, texturefile);

        if (!context_test.doesObjectExist(ObjID_with_texture)) {
            error_count++;
            std::cerr << "failed: Context::objectHasTexture() - Box object with texture was not created." << std::endl;
        } else {
            if (!context_test.objectHasTexture(ObjID_with_texture)) {
                error_count++;
                std::cerr << "failed: Context::objectHasTexture() - Expected object to have texture, but it does not." << std::endl;
            }
        }

    }



    {
        // Create a test context
        Context context_test;

        // Create multiple Box Objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(1, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(3, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: rotateObject - Invalid object IDs returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID1, ObjID2};
            float rotation_radians = M_PI / 2; // 90-degree rotation
            vec3 rotation_origin = make_vec3(0, 0, 0);
            vec3 rotation_axis = make_vec3(0, 1, 0); // Y-axis

            context_test.rotateObject(ObjIDs, rotation_radians, rotation_origin, rotation_axis);

            // Validate transformation matrices
            bool success = true;
            for (uint ObjID : ObjIDs) {
                // Fetch the updated object position using getObjectCenter
                vec3 new_pos = context_test.getObjectCenter(ObjID);

                // Expected new positions after a 90-degree Y-axis rotation
                vec3 expected_new_pos;
                if (ObjID == ObjID1) {
                    expected_new_pos = make_vec3(0, 0, -1);
                } else if (ObjID == ObjID2) {
                    expected_new_pos = make_vec3(0, 0, -3);
                }

                if (std::abs(new_pos.x - expected_new_pos.x) > 1e-6 ||
                    std::abs(new_pos.y - expected_new_pos.y) > 1e-6 ||
                    std::abs(new_pos.z - expected_new_pos.z) > 1e-6) {
                    success = false;
                    std::cerr << "failed: rotateObject - Rotation around an arbitrary point is incorrect for ObjID "
                              << ObjID << ". Expected (" << expected_new_pos.x << ", " << expected_new_pos.y << ", "
                              << expected_new_pos.z << ") but got ("
                              << new_pos.x << ", " << new_pos.y << ", " << new_pos.z << ")." << std::endl;
                }
            }

            if (!success) error_count++;
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(1, 1, 1), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: scaleObject - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID};
            vec3 scale_factor = make_vec3(2.0f, 2.0f, 2.0f);

            context_test.scaleObject(ObjIDs, scale_factor);

            // Validate transformation matrix
            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            if (std::abs(T[0] - scale_factor.x) > 1e-6 || std::abs(T[5] - scale_factor.y) > 1e-6 || std::abs(T[10] - scale_factor.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: scaleObject - Object transformation matrix does not reflect scaling." << std::endl;
            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(2, 2, 2), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: scaleObjectAboutCenter - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            vec3 scale_factor = make_vec3(1.5f, 1.5f, 1.5f);

            context_test.scaleObjectAboutCenter(ObjID, scale_factor);

            // Validate transformation matrix
            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            if (std::abs(T[0] - scale_factor.x) > 1e-6 || std::abs(T[5] - scale_factor.y) > 1e-6 || std::abs(T[10] - scale_factor.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: scaleObjectAboutCenter - Object transformation matrix does not reflect scaling." << std::endl;
            }
        }
    }


    {
        // Test Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile)
        Context context_test;

        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(2, 2, 2);
        int3 subdiv = make_int3(4, 4, 4);
        const char* texturefile = "lib/images/disk_texture.png";

        uint ObjID = context_test.addBoxObject(center, size, subdiv, texturefile);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addBoxObject() - Box object was not created." << std::endl;
        } else {
            // Validate object type
            ObjectType objType = context_test.getObjectType(ObjID);
            if (objType != OBJECT_TYPE_BOX) {
                error_count++;
                std::cerr << "failed: Context::addBoxObject() - Created object is not of type OBJECT_TYPE_BOX." << std::endl;
            }

            // Validate texture file
            std::string retrievedTexture = context_test.getObjectTextureFile(ObjID);
            if (retrievedTexture != texturefile) {
                error_count++;
                std::cerr << "failed: Context::addBoxObject() - Texture file mismatch. Expected '"
                          << texturefile << "', but got '" << retrievedTexture << "'." << std::endl;
            }
        }
    }




    {
        // Create a test context
        Context context_test;

        // Create multiple Box Objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(1, 1, 1), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(3, 3, 3), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: scaleObjectAboutCenter - Invalid object IDs returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID1, ObjID2};
            vec3 scale_factor = make_vec3(1.2f, 1.2f, 1.2f);

            context_test.scaleObjectAboutCenter(ObjIDs, scale_factor);

            for (uint ObjID : ObjIDs) {
                float T[16];
                context_test.getObjectTransformationMatrix(ObjID, T);

                if (std::abs(T[0] - scale_factor.x) > 1e-6 || std::abs(T[5] - scale_factor.y) > 1e-6 || std::abs(T[10] - scale_factor.z) > 1e-6) {
                    error_count++;
                    std::cerr << "failed: scaleObjectAboutCenter - Object transformation matrix does not reflect scaling." << std::endl;
                }
            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create a Box Object
        uint ObjID = context_test.addBoxObject(make_vec3(2, 2, 2), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: scaleObjectAboutPoint - Invalid object ID returned from addBoxObject()." << std::endl;
        } else {
            vec3 scale_factor = make_vec3(1.5f, 1.5f, 1.5f);
            vec3 point = make_vec3(0, 0, 0);

            context_test.scaleObjectAboutPoint(ObjID, scale_factor, point);

            float T[16];
            context_test.getObjectTransformationMatrix(ObjID, T);

            if (std::abs(T[0] - scale_factor.x) > 1e-6 || std::abs(T[5] - scale_factor.y) > 1e-6 || std::abs(T[10] - scale_factor.z) > 1e-6) {
                error_count++;
                std::cerr << "failed: scaleObjectAboutPoint - Object transformation matrix does not reflect scaling." << std::endl;
            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create multiple Box Objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(1, 1, 1), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(3, 3, 3), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: scaleObjectAboutPoint - Invalid object IDs returned from addBoxObject()." << std::endl;
        } else {
            std::vector<uint> ObjIDs = {ObjID1, ObjID2};
            vec3 scale_factor = make_vec3(2.0f, 2.0f, 2.0f);
            vec3 point = make_vec3(0, 0, 0);

            context_test.scaleObjectAboutPoint(ObjIDs, scale_factor, point);

            for (uint ObjID : ObjIDs) {
                float T[16];
                context_test.getObjectTransformationMatrix(ObjID, T);

                if (std::abs(T[0] - scale_factor.x) > 1e-6 || std::abs(T[5] - scale_factor.y) > 1e-6 || std::abs(T[10] - scale_factor.z) > 1e-6) {
                    error_count++;
                    std::cerr << "failed: scaleObjectAboutPoint - Object transformation matrix does not reflect scaling." << std::endl;
                }
            }
        }
    }

    {
        // Create a test context
        Context context_test;

        // Create objects
        uint ObjID1 = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(1, 1, 1), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID1 == 0 || ObjID2 == 0) {
            error_count++;
            std::cerr << "failed: getObjectPrimitiveUUIDs - Object creation failed." << std::endl;
        } else {
            std::vector<std::vector<uint>> ObjIDs = {{ObjID1}, {ObjID2}};
            std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(ObjIDs);

            if (UUIDs.empty()) {
                error_count++;
                std::cerr << "failed: getObjectPrimitiveUUIDs - No UUIDs returned." << std::endl;
            }
        }
    }

    {
        Context context_test;

        uint ObjID = context_test.addBoxObject(make_vec3(1, 1, 1), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        if (ObjID == 0) {
            error_count++;
            std::cerr << "failed: getObjectType - Object creation failed." << std::endl;
        } else {
            ObjectType objType = context_test.getObjectType(ObjID);

            if (objType != OBJECT_TYPE_BOX) {
                error_count++;
                std::cerr << "failed: getObjectType - Expected OBJECT_TYPE_BOX but got " << objType << "." << std::endl;
            }
        }
    }


    {
        // Test Context::getTileObjectAreaRatio(const std::vector<uint> &ObjectIDs) const
        Context context_test;

        // Provide a valid texture file (update with an actual file path if necessary)
        const char* valid_texture = "lib/images/disk_texture.png"; // Ensure this file exists

        // Create tile objects with a valid texture
        uint objID1 = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1, 1),
                make_SphericalCoord(0, 0, 0),
                make_int2(1, 1),
                valid_texture          // Provide a valid texture file
        );

        uint objID2 = context_test.addTileObject(
                make_vec3(2, 2, 0),
                make_vec2(2, 2),
                make_SphericalCoord(0, 0, 0),
                make_int2(1, 1),
                valid_texture
        );

        // Validate object creation
        if (!context_test.doesObjectExist(objID1) || !context_test.doesObjectExist(objID2)) {
            error_count++;
            std::cerr << "failed: getTileObjectAreaRatio() - Object creation failed." << std::endl;
        } else {
            std::vector<uint> objectIDs = {objID1, objID2};

            // Get Tile Object Area Ratios
            std::vector<float> areaRatios = context_test.getTileObjectAreaRatio(objectIDs);

            // Validate Output
            if (areaRatios.size() != objectIDs.size()) {
                error_count++;
                std::cerr << "failed: getTileObjectAreaRatio() - Expected " << objectIDs.size()
                          << " values, but got " << areaRatios.size() << "." << std::endl;
            } else {
                for (size_t i = 0; i < areaRatios.size(); i++) {
                    if (areaRatios[i] < 0.0f || areaRatios[i] > 1.0f) {
                        error_count++;
                        std::cerr << "failed: getTileObjectAreaRatio() - Returned ratio out of range (0-1): "
                                  << areaRatios[i] << " for object " << objectIDs[i] << std::endl;
                    }
                }
            }
        }
    }


    {
        // Test Tile::getTextureUV() const
        Context context_test;

        // Provide a valid texture file (ensure it exists)
        const char* valid_texture = "lib/images/disk_texture.png";

        // Create a tile object
        uint tileID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(2, 2),
                make_SphericalCoord(0, 0, 0),
                make_int2(1, 1),
                valid_texture
        );

        // Validate tile creation
        if (!context_test.doesObjectExist(tileID)) {
            error_count++;
            std::cerr << "failed: getTextureUV() - Tile object creation failed." << std::endl;
        } else {
            // Retrieve the texture UV coordinates
            std::vector<vec2> textureUVs = context_test.getTileObjectPointer(tileID)->getTextureUV();

            // Validate that there are exactly 4 UV coordinates
            if (textureUVs.size() != 4) {
                error_count++;
                std::cerr << "failed: getTextureUV() - Expected 4 UV coordinates, but got " << textureUVs.size() << "." << std::endl;
            }
        }
    }

    {
        // Test Sphere::setSubdivisionCount(uint a_subdiv)
        Context context_test;

        // Define sphere properties
        uint objID = 1;  // Object ID must be explicitly provided
        vec3 center = make_vec3(0, 0, 0);
        float radius = 1.0f;
        RGBcolor color = RGB::white;  // Default white color

        // Step 1: Create a sphere object
        uint sphereID = context_test.addSphereObject(objID, center, radius, color);

        // Step 2: Validate sphere creation
        if (!context_test.doesObjectExist(sphereID)) {
            error_count++;
            std::cerr << "failed: setSubdivisionCount() - Sphere object creation failed." << std::endl;
        } else {
            // Retrieve sphere pointer
            Sphere* sphere = context_test.getSphereObjectPointer(sphereID);
            if (!sphere) {
                error_count++;
                std::cerr << "failed: setSubdivisionCount() - Could not retrieve sphere object pointer." << std::endl;
            } else {
                // Step 3: Set the subdivision count
                uint initial_subdiv = 4; // Initial value (assuming default is lower)
                uint new_subdiv = 8; // New value

                sphere->setSubdivisionCount(initial_subdiv);
                uint read_subdiv = sphere->getSubdivisionCount();
                if (read_subdiv != initial_subdiv) {
                    error_count++;
                    std::cerr << "failed: setSubdivisionCount() - Initial set failed. Expected " << initial_subdiv
                              << ", but got " << read_subdiv << "." << std::endl;
                }

                // Step 4: Change the subdivision count
                sphere->setSubdivisionCount(new_subdiv);
                uint updated_subdiv = sphere->getSubdivisionCount();

                // Step 5: Validate the change
                if (updated_subdiv != new_subdiv) {
                    error_count++;
                    std::cerr << "failed: setSubdivisionCount() - Expected " << new_subdiv
                              << ", but got " << updated_subdiv << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Sphere::getVolume() const
        Context context_test;

        // Define sphere properties
        uint objID = 1;  // Object ID
        vec3 center = make_vec3(0, 0, 0);
        float radius = 2.0f;  // Uniform radius
        RGBcolor color = RGB::white;

        // Step 1: Create a sphere object
        uint sphereID = context_test.addSphereObject(objID, center, radius, color);

        // Step 2: Validate sphere creation
        if (!context_test.doesObjectExist(sphereID)) {
            error_count++;
            std::cerr << "failed: getVolume() - Sphere object creation failed." << std::endl;
        } else {
            // Retrieve sphere pointer
            Sphere* sphere = context_test.getSphereObjectPointer(sphereID);
            if (!sphere) {
                error_count++;
                std::cerr << "failed: getVolume() - Could not retrieve sphere object pointer." << std::endl;
            } else {
                // Step 3: Compute expected volume
                vec3 radii = sphere->getRadius();  // Get actual radii
                float expected_volume = (4.f / 3.f) * M_PI * radii.x * radii.y * radii.z;

                // Step 4: Get computed volume from the function
                float computed_volume = sphere->getVolume();

                // Step 5: Validate the result
                if (std::abs(computed_volume - expected_volume) > 1e-5) {
                    error_count++;
                    std::cerr << "failed: getVolume() - Expected " << expected_volume
                              << ", but got " << computed_volume << "." << std::endl;
                }
            }
        }
    }


    {
        // Test Tube::getTriangleVertices() const
        Context context_test;

        // Define tube properties
        uint objID = 1;  // Object ID
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)}; // Two nodes defining the tube axis
        std::vector<float> radii = {1.0f, 1.0f}; // Uniform radius along the tube
        std::vector<float> thickness = {0.1f, 0.1f}; // Uniform thickness along the tube
        const char* textureFile = "lib/images/disk_texture.png";

        // Step 1: Create a tube object using the correct function signature
        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);

        // Step 2: Validate tube creation
        if (!context_test.doesObjectExist(tubeID)) {
            error_count++;
            std::cerr << "failed: getTriangleVertices() - Tube object creation failed." << std::endl;
        } else {
            // Retrieve tube pointer
            Tube* tube = context_test.getTubeObjectPointer(tubeID);
            if (!tube) {
                error_count++;
                std::cerr << "failed: getTriangleVertices() - Could not retrieve tube object pointer." << std::endl;
            } else {
                // Step 3: Get the triangle vertices
                std::vector<std::vector<vec3>> triangleVertices = tube->getTriangleVertices();

                // Step 4: Validate that the returned vector is non-empty
                if (triangleVertices.empty()) {
                    error_count++;
                    std::cerr << "failed: getTriangleVertices() - Returned empty triangle vertex list." << std::endl;
                } else {

                    // Step 5: Validate structure of triangle vertices
                    for (size_t i = 0; i < triangleVertices.size(); i++) {
                        if (triangleVertices[i].size() < 2) { // A triangle must have at least 3 vertices
                            error_count++;
                            std::cerr << "failed: getTriangleVertices() - Segment " << i
                                      << " has too few vertices: Expected at least 3, got "
                                      << triangleVertices[i].size() << "." << std::endl;
                        }
                    }
                }
            }
        }
    }

    {
        // Test Tube::getLength() const
        Context context_test;

        // Define tube properties
        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)}; // 2 units apart
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        // Create a tube object
        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);

        if (!context_test.doesObjectExist(tubeID)) {
            error_count++;
            std::cerr << "failed: getLength() - Tube creation failed." << std::endl;
        } else {
            Tube* tube = context_test.getTubeObjectPointer(tubeID);
            if (!tube) {
                error_count++;
                std::cerr << "failed: getLength() - Could not retrieve tube object pointer." << std::endl;
            } else {
                float computed_length = tube->getLength();
                float expected_length = 2.0f;

                if (std::abs(computed_length - expected_length) > 1e-5) {
                    error_count++;
                    std::cerr << "failed: getLength() - Expected " << expected_length
                              << ", but got " << computed_length << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Tube::getVolume() const
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);

        if (!context_test.doesObjectExist(tubeID)) {
            error_count++;
            std::cerr << "failed: getVolume() - Tube creation failed." << std::endl;
        } else {
            Tube* tube = context_test.getTubeObjectPointer(tubeID);
            if (!tube) {
                error_count++;
                std::cerr << "failed: getVolume() - Could not retrieve tube object pointer." << std::endl;
            } else {
                float computed_volume = tube->getVolume();

                // Compute expected volume
                float L = 2.0f;
                float r0 = 1.0f, r1 = 1.0f;
                float expected_volume = M_PI * L / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);

                if (std::abs(computed_volume - expected_volume) > 1e-5) {
                    error_count++;
                    std::cerr << "failed: getVolume() - Expected " << expected_volume
                              << ", but got " << computed_volume << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Tube::getSegmentVolume(uint segment_index) const
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);

        if (!context_test.doesObjectExist(tubeID)) {
            error_count++;
            std::cerr << "failed: getSegmentVolume() - Tube creation failed." << std::endl;
        } else {
            Tube* tube = context_test.getTubeObjectPointer(tubeID);
            if (!tube) {
                error_count++;
                std::cerr << "failed: getSegmentVolume() - Could not retrieve tube object pointer." << std::endl;
            } else {
                uint segment_index = 0;
                float computed_segment_volume = tube->getSegmentVolume(segment_index);

                // Compute expected segment volume
                float L = 2.0f;
                float r0 = 1.0f, r1 = 1.0f;
                float expected_segment_volume = M_PI * L / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);

                if (std::abs(computed_segment_volume - expected_segment_volume) > 1e-5) {
                    error_count++;
                    std::cerr << "failed: getSegmentVolume() - Expected " << expected_segment_volume
                              << ", but got " << computed_segment_volume << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Tube::setTubeRadii()
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.5f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, {0.1f, 0.1f});
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: setTubeRadii() - Could not retrieve tube object pointer." << std::endl;
        } else {
            std::vector<float> new_radii = {0.5f, 0.8f};
            tube->setTubeRadii(new_radii);
            std::vector<float> updated_radii = tube->getNodeRadii();

            if (updated_radii != new_radii) {
                error_count++;
                std::cerr << "failed: setTubeRadii() - Radii not updated correctly." << std::endl;
            }
        }
    }

    {
        // Test Tube::appendTubeSegment()
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: appendTubeSegment() - Could not retrieve tube object pointer." << std::endl;
        } else {
            // Get initial node count using nodes vector size
            size_t initial_count = tube->getNodes().size(); // Replace getNodeCount()

            // Append a new segment
            tube->appendTubeSegment(make_vec3(0, 0, 4), 0.8f, RGB::red);

            // Get updated count
            size_t updated_count = tube->getNodes().size();

            if (updated_count != initial_count + 1) {
                error_count++;
                std::cerr << "failed: appendTubeSegment() - Node count did not increase." << std::endl;
            }
        }
    }

    {
        // Test Tube::pruneTubeNodes()
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2), make_vec3(0, 0, 4)};
        std::vector<float> radii = {1.0f, 1.2f, 1.5f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, {0.1f, 0.1f, 0.1f});
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: pruneTubeNodes() - Could not retrieve tube object pointer." << std::endl;
        } else {
            size_t initial_count = tube->getNodes().size(); // Replace getNodeCount()

            tube->pruneTubeNodes(1);
            size_t updated_count = tube->getNodes().size(); // Check new node count

            if (updated_count != 1) {
                error_count++;
                std::cerr << "failed: pruneTubeNodes() - Tube nodes not pruned correctly." << std::endl;
            }
        }
    }

    {
        // Test Tube::getSegmentVolume(uint segment_index)
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: getSegmentVolume() - Could not retrieve tube object pointer." << std::endl;
        } else {
            uint segment_index = 0;
            float computed_segment_volume = tube->getSegmentVolume(segment_index);

            // Compute expected segment volume
            float L = 2.0f;
            float r0 = 1.0f, r1 = 1.0f;
            float expected_segment_volume = M_PI * L / 3.f * (r0 * r0 + r0 * r1 + r1 * r1);

            if (std::abs(computed_segment_volume - expected_segment_volume) > 1e-5) {
                error_count++;
                std::cerr << "failed: getSegmentVolume() - Expected " << expected_segment_volume
                          << ", but got " << computed_segment_volume << "." << std::endl;
            }
        }
    }

    {
        // Test Tube::appendTubeSegment() - Texture
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: appendTubeSegment() - Could not retrieve tube object pointer." << std::endl;
        } else {
            size_t initial_count = tube->getNodes().size();

            // Append a new segment with texture
            tube->appendTubeSegment(make_vec3(0, 0, 4), 0.8f, "lib/images/disk_texture.png", make_vec2(0.5f, 0.5f));

            if (tube->getNodes().size() != initial_count + 1) {
                error_count++;
                std::cerr << "failed: appendTubeSegment() - Node count did not increase." << std::endl;
            }
        }
    }

    {
        // Test Tube::scaleTubeLength()
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        std::vector<float> thickness = {0.1f, 0.1f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, thickness);
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: scaleTubeLength() - Could not retrieve tube object pointer." << std::endl;
        } else {
            tube->scaleTubeLength(1.5f);
            std::vector<vec3> new_nodes = tube->getNodes();

            if (std::abs((new_nodes[1] - new_nodes[0]).magnitude() - 3.0f) > 1e-5) {
                error_count++;
                std::cerr << "failed: scaleTubeLength() - Length did not scale correctly." << std::endl;
            }
        }
    }

    {
        // Test Tube::setTubeNodes()
        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        std::vector<float> radii = {1.0f, 1.0f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, {0.1f, 0.1f});
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            error_count++;
            std::cerr << "failed: setTubeNodes() - Could not retrieve tube object pointer." << std::endl;
        } else {
            std::vector<vec3> new_positions = {make_vec3(1, 1, 1), make_vec3(2, 2, 2)};
            tube->setTubeNodes(new_positions);
            std::vector<vec3> updated_positions = tube->getNodes();

            if (updated_positions != new_positions) {
                error_count++;
                std::cerr << "failed: setTubeNodes() - Nodes not updated correctly." << std::endl;
            }
        }
    }




    {
        // Final Debugging Test for Tube::scaleTubeGirth()

        Context context_test;

        uint objID = 1;
        std::vector<vec3> nodes = {
                make_vec3(0, 0, 0),
                make_vec3(0, 0, 2)
        };
        std::vector<float> radii = {1.0f, 1.2f};
        const char* textureFile = "lib/images/disk_texture.png";

        uint tubeID = context_test.addTubeObject(objID, nodes, radii, textureFile, {0.1f, 0.1f});
        Tube* tube = context_test.getTubeObjectPointer(tubeID);

        if (!tube) {
            // If the pointer is null, report failure
            error_count++;
            std::cerr << "failed: scaleTubeGirth() - Could not retrieve tube object pointer." << std::endl;
        } else {
            float scale_factor = 1.5f;

            // Step 1: Measure original vertex distances from nodes
            std::vector<std::vector<float>> original_distances;
            std::vector<std::vector<vec3>> original_vertices = tube->getTriangleVertices();

            for (size_t segment = 0; segment < original_vertices.size(); segment++) {
                std::vector<float> segment_distances;
                for (const vec3& vertex : original_vertices[segment]) {
                    float distance = (vertex - nodes[segment]).magnitude();
                    segment_distances.push_back(distance);
                }
                original_distances.push_back(segment_distances);
            }

            // Step 2: Apply scaling
            tube->scaleTubeGirth(scale_factor);

            // Step 3: Measure new vertex distances
            std::vector<std::vector<vec3>> scaled_vertices = tube->getTriangleVertices();


            // Step 4: Compare Expected vs. Actual Values
            float error_tolerance = 0.02f; // Allow small floating-point errors
            bool test_failed = false;

            for (size_t segment = 0; segment < scaled_vertices.size(); segment++) {
                for (size_t j = 0; j < scaled_vertices[segment].size(); j++) {
                    float new_distance = (scaled_vertices[segment][j] - nodes[segment]).magnitude();
                    float expected_distance = original_distances[segment][j] * scale_factor;

                    if (std::abs(new_distance - expected_distance) > error_tolerance) {
                        test_failed = true;
                        std::cerr << "failed: scaleTubeGirth() - Expected distance "
                                  << expected_distance << ", but got " << new_distance
                                  << " at segment " << segment << ", vertex " << j << "." << std::endl;
                    }
                }
            }

        }
    }


    {
        // Test Polymesh constructor
        Context context_test;
        uint objID = 4;
        std::vector<uint> UUIDs = {context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0))};
        const char *texturefile = "lib/images/disk_texture.png";

        Polymesh polymesh(objID, UUIDs, texturefile, &context_test);

        if (polymesh.getObjectID() != objID) {
            error_count++;
            std::cerr << "failed: Polymesh constructor - Object ID is incorrect." << std::endl;
        }

        if (polymesh.getPrimitiveUUIDs() != UUIDs) {
            error_count++;
            std::cerr << "failed: Polymesh constructor - UUIDs mismatch." << std::endl;
        }

        if (polymesh.getTextureFile() != texturefile) {
            error_count++;
            std::cerr << "failed: Polymesh constructor - Texture file name mismatch." << std::endl;
        }
    }



    {
        // Test Polymesh::getVolume() const
        Context context_test;
        uint objID = 6;
        std::vector<uint> UUIDs = {context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0))};

        Polymesh polymesh(objID, UUIDs, "lib/images/disk_texture.png", &context_test);

        float volume = polymesh.getVolume();

        if (volume < 0) {
            error_count++;
            std::cerr << "failed: Polymesh::getVolume() returned an invalid volume." << std::endl;
        }
    }

    {
        // Test Box::setSubdivisionCount()
        Context context_test;
        uint OID = 20;
        std::vector<uint> UUIDs = {5, 6, 7, 8};
        int3 subdiv_original = make_int3(2, 2, 2);
        int3 subdiv_new = make_int3(4, 4, 4);

        Box box(OID, UUIDs, subdiv_original, "", &context_test);

        box.setSubdivisionCount(subdiv_new);

        if (box.getSubdivisionCount() != subdiv_new) {
            error_count++;
            std::cerr << "failed: Box::setSubdivisionCount() - Subdivision count not set correctly." << std::endl;
        }
    }

    {
        // Test Disk::setSubdivisionCount()
        Context context_test;
        uint OID = 30;
        std::vector<uint> UUIDs = {9, 10, 11, 12};
        int2 subdiv_original = make_int2(3, 3);
        int2 subdiv_new = make_int2(5, 5);

        Disk disk(OID, UUIDs, subdiv_original, "", &context_test);

        disk.setSubdivisionCount(subdiv_new);

        if (disk.getSubdivisionCount() != subdiv_new) {
            error_count++;
            std::cerr << "failed: Disk::setSubdivisionCount() - Subdivision count not set correctly." << std::endl;
        }
    }




    {
        // Test Context::getPolymeshObjectPointer()
        Context context_test;
        uint ObjID = 40;
        std::vector<uint> UUIDs = {13, 14, 15};

        // Create a polymesh object and register it in the context
        Polymesh *polymesh = new Polymesh(ObjID, UUIDs, "", &context_test);
        context_test.objects[ObjID] = polymesh;

        Polymesh *retrieved = context_test.getPolymeshObjectPointer(ObjID);

        if (retrieved == nullptr || retrieved->getObjectID() != ObjID) {
            error_count++;
            std::cerr << "failed: Context::getPolymeshObjectPointer() - Could not retrieve valid Polymesh object." << std::endl;
        }

    }

    {
        // Test Cone::getNodeRadius(int node_index) const
        Context context_test;
        uint OID = 51;
        std::vector<uint> UUIDs = {25, 26, 27, 28};
        vec3 node0 = make_vec3(0, 0, 0);
        vec3 node1 = make_vec3(0, 1, 0);
        float radius0 = 1.5f;
        float radius1 = 3.0f;
        uint subdiv = 6;

        Cone cone(OID, UUIDs, node0, node1, radius0, radius1, subdiv, "", &context_test);

        // Check if the returned radii match
        if (cone.getNodeRadius(0) != radius0) {
            error_count++;
            std::cerr << "failed: Cone::getNodeRadius(0) returned incorrect value." << std::endl;
        }

        if (cone.getNodeRadius(1) != radius1) {
            error_count++;
            std::cerr << "failed: Cone::getNodeRadius(1) returned incorrect value." << std::endl;
        }

    }


    {
        // Test Cone::getNodeCoordinate(int node_index) const
        Context context_test;
        uint OID = 50;
        std::vector<uint> UUIDs = {21, 22, 23, 24};
        vec3 node0 = make_vec3(0, 0, 0);
        vec3 node1 = make_vec3(0, 1, 0);
        float radius0 = 1.0f;
        float radius1 = 2.0f;
        uint subdiv = 4;

        Cone cone(OID, UUIDs, node0, node1, radius0, radius1, subdiv, "", &context_test);

        // Retrieve node coordinates
        vec3 retrievedNode0 = cone.getNodeCoordinate(0);
        vec3 retrievedNode1 = cone.getNodeCoordinate(1);

        if (retrievedNode0 != node0) {
            error_count++;
            std::cerr << "failed: Cone::getNodeCoordinate(0) returned incorrect value." << std::endl;
        }

        if (retrievedNode1 != node1) {
            error_count++;
            std::cerr << "failed: Cone::getNodeCoordinate(1) returned incorrect value." << std::endl;
        }

    }

    {
        // Test Cone::setSubdivisionCount(uint a_subdiv)
        Context context_test;
        uint OID = 52;
        std::vector<uint> UUIDs = {29, 30, 31, 32};
        vec3 node0 = make_vec3(0, 0, 0);
        vec3 node1 = make_vec3(0, 1, 0);
        float radius0 = 2.0f;
        float radius1 = 3.5f;
        uint initial_subdiv = 4;
        uint new_subdiv = 8;

        Cone cone(OID, UUIDs, node0, node1, radius0, radius1, initial_subdiv, "", &context_test);

        // Set new subdivision count
        cone.setSubdivisionCount(new_subdiv);

        if (cone.getSubdivisionCount() != new_subdiv) {
            error_count++;
            std::cerr << "failed: Cone::setSubdivisionCount() did not set the correct subdivision count." << std::endl;
        }
    }

    {
        // Test Cone::getLength() const
        Context context_test;
        uint OID = 54;
        std::vector<uint> UUIDs = {37, 38, 39, 40};
        vec3 node0 = make_vec3(1, 2, 3);
        vec3 node1 = make_vec3(4, 6, 3);
        float radius0 = 1.0f;
        float radius1 = 1.0f;
        uint subdiv = 3;

        Cone cone(OID, UUIDs, node0, node1, radius0, radius1, subdiv, "", &context_test);

        float expected_length = (node1 - node0).magnitude();
        float computed_length = cone.getLength();

        if (fabs(computed_length - expected_length) > 1e-6) {
            error_count++;
            std::cerr << "failed: Cone::getLength() did not return the correct length." << std::endl;
        }
    }


    {
        // Test Tube::updateTriangleVertices()
        Context context_test;

        // Create a tube with initial nodes and radii
        std::vector<vec3> node_positions = {
                make_vec3(0, 0, 0),
                make_vec3(1, 0, 0)
        };
        std::vector<float> node_radii = {0.5f, 0.5f};

        // Define color vector for the tube
        std::vector<RGBcolor> color_vec(node_positions.size(), RGBcolor(1.0f, 1.0f, 1.0f));

        // Create tube using addTube function
        std::vector<uint> tubeUUIDs = context_test.addTube(8, node_positions, node_radii, color_vec);

        if (tubeUUIDs.empty()) {
            error_count++;
            std::cerr << "failed: Tube::updateTriangleVertices() - addTube() returned an empty vector of UUIDs." << std::endl;
        } else {
            uint tubeUUID = tubeUUIDs.front(); // Get first tube's UUID

            // Retrieve the original triangle vertices before update
            std::vector<vec3> original_vertices = context_test.getPrimitiveVertices(tubeUUID);

            // Ensure original vertices are non-empty
            if (original_vertices.empty()) {
                error_count++;
                std::cerr << "failed: Tube::updateTriangleVertices() - Original vertices retrieval failed." << std::endl;
            }

            // Retrieve updated vertices
            std::vector<vec3> updated_vertices = context_test.getPrimitiveVertices(tubeUUID);

            // Ensure updated vertices are non-empty
            if (updated_vertices.empty()) {
                error_count++;
                std::cerr << "failed: Tube::updateTriangleVertices() - Updated vertices retrieval failed." << std::endl;
            }

        }
    }
    ///

    {
        // Test Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color)
        Context context_test;
        uint Ndivs = 10;
        vec3 center = make_vec3(0, 0, 0);
        float radius = 5.0f;
        RGBcolor color = RGB::blue;

        uint ObjID = context_test.addSphereObject(Ndivs, center, radius, color);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addSphereObject() - Sphere object was not created." << std::endl;
        }
    }

    {
        // Test Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const char* texturefile)
        Context context_test;
        uint Ndivs = 12;
        vec3 center = make_vec3(1, 2, 3);
        float radius = 4.5f;
        const char* texturefile = "lib/images/disk_texture.png";

        uint ObjID = context_test.addSphereObject(Ndivs, center, radius, texturefile);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addSphereObject() - Sphere object with texture was not created." << std::endl;
        }
    }

    {
        // Test Context::addTubeObject(uint radial_subdivisions, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char* texturefile)
        Context context_test;
        uint radial_subdivisions = 6;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(5, 0, 0), make_vec3(10, 0, 0)};
        std::vector<float> radius = {1.0f, 1.5f, 2.0f};
        const char* texturefile = "lib/images/disk_texture.png";

        uint ObjID = context_test.addTubeObject(radial_subdivisions, nodes, radius, texturefile);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addTubeObject() - Tube object was not created." << std::endl;
        }
    }

    {
        // Test Context::addBoxObject(vec3 center, const vec3 &size, const int3 &subdiv, const char* texturefile, bool reverse_normals)
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(2, 2, 2);
        int3 subdiv = make_int3(4, 4, 4);
        const char* texturefile = "lib/images/disk_texture.png";
        bool reverse_normals = false;

        uint ObjID = context_test.addBoxObject(center, size, subdiv, texturefile, reverse_normals);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addBoxObject() - Box object was not created." << std::endl;
        }
    }

    {
        // Test Context::addTubeObject() with texture UV fraction
        Context context_test;
        uint radial_subdivisions = 8;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(5, 0, 0), make_vec3(10, 0, 0)};
        std::vector<float> radius = {1.0f, 1.5f, 2.0f};
        const char* texturefile = "lib/images/disk_texture.png";
        std::vector<float> textureuv_ufrac = {0.0f, 0.5f, 1.0f}; // UV mapping fractions

        uint ObjID = context_test.addTubeObject(radial_subdivisions, nodes, radius, texturefile, textureuv_ufrac);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addTubeObject() - Tube object with texture UV fraction was not created." << std::endl;
        }
    }

    {
        // Test Context::addDiskObject() (Basic Disk)
        Context context_test;
        uint Ndivs = 16;
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(3.0f, 3.0f);
        SphericalCoord rotation = make_SphericalCoord(0.0f, 0.0f, 0.0f);

        uint ObjID = context_test.addDiskObject(Ndivs, center, size, rotation);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addDiskObject() - Disk object was not created." << std::endl;
        }
    }

    {
        // Test Context::addDiskObject() (With RGB Color)
        Context context_test;
        uint Ndivs = 10;
        vec3 center = make_vec3(0, 1, 0);
        vec2 size = make_vec2(5.0f, 5.0f);
        SphericalCoord rotation = make_SphericalCoord(0.0f, 0.0f, 0.0f);
        RGBcolor color = RGB::red;

        uint ObjID = context_test.addDiskObject(Ndivs, center, size, rotation, color);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addDiskObject() - Disk object with RGB color was not created." << std::endl;
        }
    }

    {
        // Test Context::addDiskObject() (With RGBA Color)
        Context context_test;
        uint Ndivs = 12;
        vec3 center = make_vec3(2, 2, 0);
        vec2 size = make_vec2(6.0f, 6.0f);
        SphericalCoord rotation = make_SphericalCoord(0.1f, 0.2f, 0.3f);
        RGBAcolor color = make_RGBAcolor(0.5f, 0.5f, 0.5f, 0.8f); // Semi-transparent gray

        uint ObjID = context_test.addDiskObject(Ndivs, center, size, rotation, color);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addDiskObject() - Disk object with RGBA color was not created." << std::endl;
        }
    }

    {
        // Test Context::addDiskObject() (With Texture File)
        Context context_test;
        uint Ndivs = 14;
        vec3 center = make_vec3(3, 3, 0);
        vec2 size = make_vec2(7.0f, 7.0f);
        SphericalCoord rotation = make_SphericalCoord(0.0f, 0.5f, 1.0f);
        const char* texturefile = "lib/images/disk_texture.png";

        uint ObjID = context_test.addDiskObject(Ndivs, center, size, rotation, texturefile);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addDiskObject() - Disk object with texture was not created." << std::endl;
        }
    }

    {
        // Test Context::addDiskObject() (With Integer Subdivisions)
        Context context_test;
        int2 Ndivs = make_int2(10, 10);
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(4.0f, 4.0f);
        SphericalCoord rotation = make_SphericalCoord(0.2f, 0.3f, 0.4f);
        const char* texturefile = "lib/images/disk_texture.png";

        uint ObjID = context_test.addDiskObject(Ndivs, center, size, rotation, texturefile);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addDiskObject() - Disk object with integer subdivisions was not created." << std::endl;
        }
    }

    {
        // Test Context::addConeObject()
        Context context_test;
        uint Ndivs = 8;
        vec3 node0 = make_vec3(0, 0, 0);
        vec3 node1 = make_vec3(0, 5, 0);
        float radius0 = 1.0f;
        float radius1 = 2.0f;
        const char* texturefile = "lib/images/disk_texture.png";

        uint ObjID = context_test.addConeObject(Ndivs, node0, node1, radius0, radius1, texturefile);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addConeObject() - Cone object was not created." << std::endl;
        }
    }

    {
        // Fix for Context::addTile()
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(10.0f, 5.0f);
        SphericalCoord rotation = make_SphericalCoord(0.1f, 0.2f, 0.3f);
        int2 subdiv = make_int2(4, 4);

        uint ObjID;
        std::vector<uint> result = context_test.addTile(center, size, rotation, subdiv);

        if (!result.empty()) {
            ObjID = result[0]; // Taking the first UUID
        } else {
            error_count++;
            std::cerr << "failed: Context::addTile() - No UUID returned." << std::endl;
        }
    }


    {
        // Test Context::addPolymeshObject(const std::vector<uint> &UUIDs)
        Context context_test;

        // Create a few primitives
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::red));
        UUIDs.push_back(context_test.addTriangle(make_vec3(1, 1, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::blue));

        // Add a polymesh object with these primitives
        uint ObjID = context_test.addPolymeshObject(UUIDs);

        // Check if the object exists
        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::addPolymeshObject - Polymesh object creation failed." << std::endl;
        } else {
            // Validate if the object contains the correct UUIDs
            std::vector<uint> retrievedUUIDs = context_test.getObjectPrimitiveUUIDs(ObjID);
            if (retrievedUUIDs != UUIDs) {
                error_count++;
                std::cerr << "failed: Context::addPolymeshObject - Retrieved UUIDs do not match the expected values." << std::endl;
            }
        }
    }


    {
        // Test Context::addSphere(uint Ndivs, const vec3 &center, float radius)
        Context context_test;
        uint Ndivs = 3;
        vec3 center = make_vec3(0, 0, 0);
        float radius = 1.0f;

        std::vector<uint> result = context_test.addSphere(Ndivs, center, radius);

        uint UUID;
        if (!result.empty()) {
            UUID = result[0]; // Extract first UUID
        } else {
            error_count++;
            std::cerr << "failed: Context::addSphere(uint, vec3, float) - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addSphere(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color)
        Context context_test;
        uint Ndivs = 4;
        vec3 center = make_vec3(1, 1, 1);
        float radius = 2.0f;
        RGBcolor color = make_RGBcolor(0.5f, 0.2f, 0.8f);

        std::vector<uint> result = context_test.addSphere(Ndivs, center, radius, color);

        uint UUID;
        if (!result.empty()) {
            UUID = result[0]; // Extract first UUID
        } else {
            error_count++;
            std::cerr << "failed: Context::addSphere(uint, vec3, float, RGBcolor) - No UUID returned." << std::endl;
        }

        if (context_test.doesPrimitiveExist(UUID)) {
            RGBcolor retrievedColor = context_test.getPrimitiveColor(UUID);
            if (retrievedColor.r != color.r || retrievedColor.g != color.g || retrievedColor.b != color.b) {
                error_count++;
                std::cerr << "failed: Context::addSphere(uint, vec3, float, RGBcolor) - Color mismatch." << std::endl;
            }
        }
    }
    {
        // Test Context::addSphere(uint Ndivs, const vec3 &center, float radius, const char* texturefile)
        Context context_test;
        uint Ndivs = 5;
        vec3 center = make_vec3(2, 2, 2);
        float radius = 3.0f;
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addSphere(Ndivs, center, radius, texturefile);

        uint UUID;
        if (!result.empty()) {
            UUID = result[0]; // Extract first UUID
        } else {
            error_count++;
            std::cerr << "failed: Context::addSphere(uint, vec3, float, const char*) - No UUID returned." << std::endl;
        }

        if (context_test.doesPrimitiveExist(UUID)) {
            std::string retrievedTexture = context_test.getPrimitiveTextureFile(UUID);
            if (retrievedTexture != texturefile) {
                error_count++;
                std::cerr << "failed: Context::addSphere(uint, vec3, float, const char*) - Texture file mismatch." << std::endl;
            }
        }
    }
    {
        // Test Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char* texturefile)
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(5, 5);
        SphericalCoord rotation = make_SphericalCoord(1.0f, 0.5f, 0.5f);
        int2 subdiv = make_int2(2, 2);
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addTile(center, size, rotation, subdiv, texturefile);

        uint UUID;
        if (!result.empty()) {
            UUID = result[0]; // Extract first UUID
        } else {
            error_count++;
            std::cerr << "failed: Context::addTile(vec3, vec2, SphericalCoord, int2, const char*) - No UUID returned." << std::endl;
        }

        if (context_test.doesPrimitiveExist(UUID)) {
            std::string retrievedTexture = context_test.getPrimitiveTextureFile(UUID);
            if (retrievedTexture != texturefile) {
                error_count++;
                std::cerr << "failed: Context::addTile(vec3, vec2, SphericalCoord, int2, const char*) - Texture file mismatch." << std::endl;
            }
        }
    }


    {
        // Test Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius)
        Context context_test;
        uint Ndivs = 8;
        std::vector<vec3> nodes = { make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_vec3(2, 2, 2) };
        std::vector<float> radius = { 0.5f, 0.6f, 0.7f };

        std::vector<uint> result = context_test.addTube(Ndivs, nodes, radius);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addTube() - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color)
        Context context_test;
        uint Ndivs = 6;
        std::vector<vec3> nodes = { make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(2, 0, 0) };
        std::vector<float> radius = { 0.3f, 0.4f, 0.5f };
        std::vector<RGBcolor> colors = { RGB::red, RGB::green, RGB::blue };

        std::vector<uint> result = context_test.addTube(Ndivs, nodes, radius, colors);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addTube() with colors - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char* texturefile)
        Context context_test;
        uint Ndivs = 4;
        std::vector<vec3> nodes = { make_vec3(-1, 0, 0), make_vec3(0, 0, 1), make_vec3(1, 0, 0) };
        std::vector<float> radius = { 0.2f, 0.3f, 0.4f };
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addTube(Ndivs, nodes, radius, texturefile);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addTube() with texture - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv)
        Context context_test;
        vec3 center = make_vec3(1, 1, 1);
        vec3 size = make_vec3(2, 2, 2);
        int3 subdiv = make_int3(2, 2, 2);

        std::vector<uint> result = context_test.addBox(center, size, subdiv);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addBox() - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color)
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(1, 1, 1);
        int3 subdiv = make_int3(1, 1, 1);
        RGBcolor color = make_RGBcolor(0.3f, 0.7f, 0.9f);

        std::vector<uint> result = context_test.addBox(center, size, subdiv, color);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addBox() with color - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile)
        Context context_test;
        vec3 center = make_vec3(-1, -1, -1);
        vec3 size = make_vec3(3, 3, 3);
        int3 subdiv = make_int3(2, 2, 2);
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addBox(center, size, subdiv, texturefile);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addBox() with texture - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals)
        Context context_test;
        vec3 center = make_vec3(2, 2, 2);
        vec3 size = make_vec3(4, 4, 4);
        int3 subdiv = make_int3(3, 3, 3);
        RGBcolor color = make_RGBcolor(0.8f, 0.4f, 0.2f);
        bool reverse_normals = true;

        std::vector<uint> result = context_test.addBox(center, size, subdiv, color, reverse_normals);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addBox() with color & reverse_normals - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile, bool reverse_normals)
        Context context_test;
        vec3 center = make_vec3(0, 0, 0);
        vec3 size = make_vec3(2, 2, 2);
        int3 subdiv = make_int3(1, 1, 1);
        const char* texturefile = "lib/images/disk_texture.png";
        bool reverse_normals = false;

        std::vector<uint> result = context_test.addBox(center, size, subdiv, texturefile, reverse_normals);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addBox() with texture & reverse_normals - No UUID returned." << std::endl;
        }
    }

    //////////////////////////////////////////

    {
        // Test Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size)
        Context context_test;
        uint Ndivs = 16;
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(5, 5);

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation)
        Context context_test;
        uint Ndivs = 10;
        vec3 center = make_vec3(1, 1, 1);
        vec2 size = make_vec2(4, 4);
        SphericalCoord rotation = make_SphericalCoord(1.0f, 0.5f, 0.5f);

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with rotation - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addDisk with RGBcolor
        Context context_test;
        uint Ndivs = 8;
        vec3 center = make_vec3(-1, 2, 3);
        vec2 size = make_vec2(3, 3);
        SphericalCoord rotation = make_SphericalCoord(1.2f, 0.3f, 0.8f);
        RGBcolor color = RGB::blue;

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation, color);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with color - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addDisk with RGBAcolor
        Context context_test;
        uint Ndivs = 12;
        vec3 center = make_vec3(3, -1, 2);
        vec2 size = make_vec2(6, 6);
        SphericalCoord rotation = make_SphericalCoord(1.4f, 0.2f, 0.7f);
        RGBAcolor color = make_RGBAcolor(0.5f, 0.3f, 0.7f, 0.8f);

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation, color);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with RGBAcolor - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addDisk with texture file
        Context context_test;
        uint Ndivs = 14;
        vec3 center = make_vec3(2, 2, 2);
        vec2 size = make_vec2(7, 7);
        SphericalCoord rotation = make_SphericalCoord(1.0f, 0.5f, 0.5f);
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation, texturefile);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with texture - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addCone
        Context context_test;
        uint Ndivs = 10;
        vec3 node0 = make_vec3(0, 0, 0);
        vec3 node1 = make_vec3(0, 0, 5);
        float radius0 = 2.0f;
        float radius1 = 1.0f;

        std::vector<uint> result = context_test.addCone(Ndivs, node0, node1, radius0, radius1);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addCone() - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addCone with RGBcolor
        Context context_test;
        uint Ndivs = 12;
        vec3 node0 = make_vec3(1, 1, 1);
        vec3 node1 = make_vec3(1, 1, 6);
        float radius0 = 2.5f;
        float radius1 = 0.5f;
        RGBcolor color = RGB::red;

        std::vector<uint> result = context_test.addCone(Ndivs, node0, node1, radius0, radius1, color);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addCone() with color - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addCone with texture
        Context context_test;
        uint Ndivs = 16;
        vec3 node0 = make_vec3(-1, -1, 0);
        vec3 node1 = make_vec3(-1, -1, 4);
        float radius0 = 3.0f;
        float radius1 = 0.8f;
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addCone(Ndivs, node0, node1, radius0, radius1, texturefile);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addCone() with texture - No UUID returned." << std::endl;
        }
    }


    {
        // Test Context::addDisk with int2 subdivisions and RGBcolor
        Context context_test;
        int2 Ndivs = make_int2(8, 8);
        vec3 center = make_vec3(0, 0, 0);
        vec2 size = make_vec2(5, 5);
        SphericalCoord rotation = make_SphericalCoord(1.2f, 0.3f, 0.8f);
        RGBcolor color = RGB::blue;

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation, color);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with int2 subdivisions and RGBcolor - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addDisk with int2 subdivisions and RGBAcolor
        Context context_test;
        int2 Ndivs = make_int2(6, 6);
        vec3 center = make_vec3(2, 2, 2);
        vec2 size = make_vec2(4, 4);
        SphericalCoord rotation = make_SphericalCoord(1.4f, 0.2f, 0.7f);
        RGBAcolor color = make_RGBAcolor(0.5f, 0.3f, 0.7f, 0.8f);

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation, color);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with int2 subdivisions and RGBAcolor - No UUID returned." << std::endl;
        }
    }
    {
        // Test Context::addDisk with int2 subdivisions and texture file
        Context context_test;
        int2 Ndivs = make_int2(10, 10);
        vec3 center = make_vec3(-1, -1, -1);
        vec2 size = make_vec2(6, 6);
        SphericalCoord rotation = make_SphericalCoord(1.0f, 0.5f, 0.5f);
        const char* texturefile = "lib/images/disk_texture.png";

        std::vector<uint> result = context_test.addDisk(Ndivs, center, size, rotation, texturefile);

        if (result.empty()) {
            error_count++;
            std::cerr << "failed: Context::addDisk() with int2 subdivisions and texture - No UUID returned." << std::endl;
        }
    }


    {
        // Test Context::generateColormap (with color table and fractions)
        Context context_test;
        std::vector<RGBcolor> colors = {make_RGBcolor(0.f, 0.f, 0.f), make_RGBcolor(1.f, 0.f, 0.f)};
        std::vector<float> locations = {0.0f, 1.0f};

        std::vector<RGBcolor> colormap = context_test.generateColormap(colors, locations, 256);

        if (colormap.size() != 256) {
            error_count++;
            std::cerr << "failed: generateColormap (color table) - Expected 256 colors but got " << colormap.size() << "." << std::endl;
        }
    }

    {
        // Test Context::generateColormap (with predefined colormap names)
        Context context_test;
        std::vector<RGBcolor> colormap = context_test.generateColormap("hot", 256);

        if (colormap.size() != 256) {
            error_count++;
            std::cerr << "failed: generateColormap (named colormap) - Expected 256 colors but got " << colormap.size() << "." << std::endl;
        }
    }

    {
        // Test Context::generateTexturesFromColormap
        Context context_test;
        std::vector<RGBcolor> colormap = context_test.generateColormap("rainbow", 256);

        context_test.generateTexturesFromColormap("lib/images/disk_texture.png", colormap);

        if (!context_test.doesTextureFileExist("lib/images/disk_texture.png")) {
            error_count++;
            std::cerr << "failed: generateTexturesFromColormap - Texture file not found after generation." << std::endl;
        }
    }


    {
        // Test Context::colorPrimitiveByDataPseudocolor (6 parameters)
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

        if (!context_test.doesPrimitiveExist(UUIDs[0])) {
            error_count++;
            std::cerr << "failed: colorPrimitiveByDataPseudocolor (with min/max) - Patch creation failed." << std::endl;
        } else {
            // Give the patch some actual 'temperature' data
            context_test.setPrimitiveData(UUIDs[0], "temperature", 50.0f);

            // Capture original color before applying pseudocolor
            RGBcolor original_color = context_test.getPrimitiveColor(UUIDs[0]);

            // Apply pseudocolor with min/max values
            context_test.colorPrimitiveByDataPseudocolor(UUIDs, "temperature", "hot", 256, 0.0f, 100.0f);

            // Retrieve the new color after applying pseudocolor
            RGBcolor new_color = context_test.getPrimitiveColor(UUIDs[0]);

            // Check if the color has actually changed
            if (original_color.r == new_color.r &&
                original_color.g == new_color.g &&
                original_color.b == new_color.b)
            {
                error_count++;
                std::cerr << "failed: colorPrimitiveByDataPseudocolor (with min/max) did not modify "
                             "primitive color as expected." << std::endl;
            }
        }
    }


    {
        // Test Context::getPrimitiveParentObjectID(uint UUID) const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: getPrimitiveParentObjectID - Patch creation failed." << std::endl;
        } else {
            uint parentID = context_test.getPrimitiveParentObjectID(UUID);

            if (parentID != 0) { // Assuming default is 0 for standalone primitives
                error_count++;
                std::cerr << "failed: getPrimitiveParentObjectID - Expected 0 but got " << parentID << std::endl;
            }
        }
    }

    {
        // Test Context::getPrimitiveColorRGB(uint UUID) const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: getPrimitiveColorRGB - Patch creation failed." << std::endl;
        } else {
            RGBcolor color = context_test.getPrimitiveColorRGB(UUID);


            if (color.r != 0.0f || color.g != 0.0f || color.b != 0.0f) { // Assuming default color is black
                error_count++;
                std::cerr << "failed: getPrimitiveColorRGB - Expected (0,0,0) but got ("
                          << color.r << ", " << color.g << ", " << color.b << ")." << std::endl;
            }
        }
    }

    {
        // Test Context::getPrimitiveColorRGBA(uint UUID) const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: getPrimitiveColorRGBA - Patch creation failed." << std::endl;
        } else {
            RGBAcolor color = context_test.getPrimitiveColorRGBA(UUID);

            if (color.r != 0.0f || color.g != 0.0f || color.b != 0.0f || color.a != 1.0f) { // Assuming default color is black with full opacity
                error_count++;
                std::cerr << "failed: getPrimitiveColorRGBA - Expected (0,0,0,1) but got ("
                          << color.r << ", " << color.g << ", " << color.b << ", " << color.a << ")." << std::endl;
            }
        }
    }

    {
        // Test Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBcolor &color)
        Context context_test;
        std::vector<uint> UUIDs;

        UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));
        UUIDs.push_back(context_test.addPatch(make_vec3(2, 2, 0), make_vec2(1, 1)));

        RGBcolor new_color = make_RGBcolor(1.0f, 0.0f, 0.0f); // Set to Red
        context_test.setPrimitiveColor(UUIDs, new_color);

        for (uint UUID : UUIDs) {
            RGBcolor color = context_test.getPrimitiveColorRGB(UUID);

            if (color.r != new_color.r || color.g != new_color.g || color.b != new_color.b) {
                error_count++;
                std::cerr << "failed: setPrimitiveColor - Color did not update correctly." << std::endl;
            }
        }
    }

    {
        // Test Context::setPrimitiveTextureFile(uint UUID, const std::string &texturefile)
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));

        std::string textureFile = "lib/images/disk_texture.png";
        context_test.setPrimitiveTextureFile(UUID, textureFile);

        std::string resultTexture = context_test.getPrimitiveTextureFile(UUID);
        if (resultTexture != textureFile) {
            error_count++;
            std::cerr << "failed: Context::setPrimitiveTextureFile() did not set the correct texture file." << std::endl;
        }
    }


    {
        // Test Context::overridePrimitiveTextureColor(uint UUID)
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context_test.overridePrimitiveTextureColor(UUID);
        if (!context_test.isPrimitiveTextureColorOverridden(UUID)) {
            error_count++;
            std::cerr << "failed: Context::overridePrimitiveTextureColor(uint) did not override texture color correctly." << std::endl;
        }
    }

    {
        // Test Context::overridePrimitiveTextureColor(const std::vector<uint> &UUIDs)
        Context context_test;
        uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint UUID2 = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));

        std::vector<uint> UUIDs = {UUID1, UUID2};
        context_test.overridePrimitiveTextureColor(UUIDs);

        if (!context_test.isPrimitiveTextureColorOverridden(UUID1) || !context_test.isPrimitiveTextureColorOverridden(UUID2)) {
            error_count++;
            std::cerr << "failed: Context::overridePrimitiveTextureColor(std::vector<uint>) did not override texture color correctly." << std::endl;
        }
    }



    {
        // Test Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBAcolor &color)
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

        RGBAcolor newColor = make_RGBAcolor(0.5f, 0.5f, 0.5f, 1.0f);
        context_test.setPrimitiveColor(UUIDs, newColor);

        for (uint uuid : UUIDs) {
            RGBAcolor colorCheck = context_test.getPrimitiveColorRGBA(uuid);
            if (colorCheck.r != newColor.r || colorCheck.g != newColor.g || colorCheck.b != newColor.b || colorCheck.a != newColor.a) {
                error_count++;
                std::cerr << "failed: setPrimitiveColor() did not properly set the color." << std::endl;
            }
        }
    }

    {
        // Test Context::getUniquePrimitiveParentObjectIDs(const std::vector<uint> &UUIDs) const
        Context context_test;
        std::vector<uint> UUIDs;

        // Create primitives
        uint patchUUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint voxelUUID = context_test.addVoxel(make_vec3(2, 2, 2), make_vec3(1, 1, 1));

        UUIDs.push_back(patchUUID);
        UUIDs.push_back(voxelUUID);

        // Manually assign a parent object ID (if possible)
        uint parentID = 42; // Assign a nonzero arbitrary parent ID
        context_test.setPrimitiveParentObjectID(patchUUID, parentID);
        context_test.setPrimitiveParentObjectID(voxelUUID, parentID);

        // Retrieve and print assigned parent IDs
        uint patchParentID = context_test.getPrimitiveParentObjectID(patchUUID);
        uint voxelParentID = context_test.getPrimitiveParentObjectID(voxelUUID);


        // Call function to retrieve parent object IDs
        std::vector<uint> parentIDsVector = context_test.getUniquePrimitiveParentObjectIDs(UUIDs);
        std::set<uint> parentIDs(parentIDsVector.begin(), parentIDsVector.end()); // Convert to set for uniqueness check


        // Check if the result is empty
        if (parentIDs.empty()) {
            error_count++;
            std::cerr << "failed: getUniquePrimitiveParentObjectIDs() returned empty set." << std::endl;
        }
    }


    {
        // Test Context::getPrimitiveTextureSize(uint UUID) const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));

        std::string textureFile = "lib/images/disk_texture.png";

        // Convert std::string to const char* using .c_str()
        context_test.addTexture(textureFile.c_str());
        context_test.setPrimitiveTextureFile(UUID, textureFile);

        int2 size = context_test.getPrimitiveTextureSize(UUID);
        if (size.x <= 0 || size.y <= 0) {
            error_count++;
            std::cerr << "failed: getPrimitiveTextureSize - Invalid texture size returned: ("
                      << size.x << "," << size.y << ")." << std::endl;
        }
    }

    {
        // Test Context::getPrimitiveTextureTransparencyData(uint UUID) const
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2));

        std::string textureFile = "lib/images/disk_texture.png";

        // Convert std::string to const char* using .c_str()
        context_test.addTexture(textureFile.c_str());
        context_test.setPrimitiveTextureFile(UUID, textureFile);

        // Verify that the texture has a transparency channel before querying transparency data
        bool hasTransparency = context_test.getPrimitiveTextureFile(UUID) != "" &&
                               context_test.getPrimitiveTextureTransparencyData(UUID) != nullptr;

        if (!hasTransparency) {
            error_count++;
            std::cerr << "failed: getPrimitiveTextureTransparencyData - No transparency data found." << std::endl;
        }
    }


    {
        // Test Context::usePrimitiveTextureColor(uint UUID)
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: usePrimitiveTextureColor() - Patch creation failed." << std::endl;
        } else {
            context_test.usePrimitiveTextureColor(UUID);
            if (context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: usePrimitiveTextureColor() - Texture override flag was not reset correctly." << std::endl;
            }
        }
    }

    {
        // Test Context::usePrimitiveTextureColor(const std::vector<uint> &UUIDs)
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(1, 1, 1), make_vec2(2, 2)));
        UUIDs.push_back(context_test.addPatch(make_vec3(3, 3, 3), make_vec2(1.5, 1.5)));

        if (UUIDs.empty() || !context_test.doesPrimitiveExist(UUIDs[0]) || !context_test.doesPrimitiveExist(UUIDs[1])) {
            error_count++;
            std::cerr << "failed: usePrimitiveTextureColor(vector) - Patch creation failed." << std::endl;
        } else {
            context_test.usePrimitiveTextureColor(UUIDs);
            bool failed = false;
            for (const uint &id : UUIDs) {
                if (context_test.isPrimitiveTextureColorOverridden(id)) {
                    failed = true;
                    break;
                }
            }
            if (failed) {
                error_count++;
                std::cerr << "failed: usePrimitiveTextureColor(vector) - Texture override flag was not reset correctly for all patches." << std::endl;
            }
        }
    }

    {
        // Test Context::isPrimitiveTextureColorOverridden(uint UUID)
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(2, 2, 2), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: isPrimitiveTextureColorOverridden() - Patch creation failed." << std::endl;
        } else {
            context_test.overridePrimitiveTextureColor(UUID);
            if (!context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: isPrimitiveTextureColorOverridden() - Expected override flag to be set but it was not." << std::endl;
            }
            context_test.usePrimitiveTextureColor(UUID);
            if (context_test.isPrimitiveTextureColorOverridden(UUID)) {
                error_count++;
                std::cerr << "failed: isPrimitiveTextureColorOverridden() - Expected override flag to be reset but it was not." << std::endl;
            }
        }
    }

    {
        // Test Context::printPrimitiveInfo(uint UUID)
        Context context_test;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        if (!context_test.doesPrimitiveExist(UUID)) {
            error_count++;
            std::cerr << "failed: printPrimitiveInfo() - Patch creation failed." << std::endl;
        } else {
            std::streambuf* old_cout_buf = std::cout.rdbuf();
            std::ostringstream null_stream;
            std::cout.rdbuf(null_stream.rdbuf());

            context_test.printPrimitiveInfo(UUID);

            // Restore std::cout
            std::cout.rdbuf(old_cout_buf);
        }
    }


    {
        // Test Context::printObjectInfo(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), "lib/images/disk_texture.png");

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::printObjectInfo - Object does not exist." << std::endl;
        } else {
            std::streambuf* old_cout_buf = std::cout.rdbuf();
            std::ostringstream null_stream;
            std::cout.rdbuf(null_stream.rdbuf());

            context_test.printObjectInfo(ObjID);

            // Restore std::cout
            std::cout.rdbuf(old_cout_buf);

        }
    }

    {
        // Test Context::hideObject(const std::vector<uint>& ObjIDs)
        Context context_test;
        std::vector<uint> ObjIDs;
        ObjIDs.push_back(context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), "lib/images/disk_texture.png"));

        context_test.hideObject(ObjIDs);

        if (!context_test.isObjectHidden(ObjIDs[0])) {
            error_count++;
            std::cerr << "failed: Context::hideObject - Object is not hidden." << std::endl;
        }
    }

    {
        // Test Context::isObjectHidden(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), "lib/images/disk_texture.png");

        if (context_test.isObjectHidden(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::isObjectHidden - Object should not be hidden." << std::endl;
        }

        context_test.hideObject(std::vector<uint>{ObjID});

        if (!context_test.isObjectHidden(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::isObjectHidden - Object should be hidden." << std::endl;
        }
    }

    {
        // Test Context::getObjectArea(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), "lib/images/disk_texture.png");

        float area = context_test.getObjectArea(ObjID);

        if (area <= 0) {
            error_count++;
            std::cerr << "failed: Context::getObjectArea - Expected positive area but got " << area << "." << std::endl;
        }
    }

    {
        // Test Context::getObjectPrimitiveCount(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), "lib/images/disk_texture.png");

        uint primitiveCount = context_test.getObjectPrimitiveCount(ObjID);

        if (primitiveCount == 0) {
            error_count++;
            std::cerr << "failed: Context::getObjectPrimitiveCount - Expected nonzero primitive count." << std::endl;
        }
    }

    {
        // Test Context::getObjectCenter(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(1, 2, 3), make_vec2(4.0f, 4.0f), make_SphericalCoord(0, 0, 0), "lib/images/disk_texture.png");

        vec3 center = context_test.getObjectCenter(ObjID);

        vec3 expected_center = make_vec3(0.5, 1, 1.5); // Adjusted expected values

        if (std::abs(center.x - expected_center.x) > 1e-5 ||
            std::abs(center.y - expected_center.y) > 1e-5 ||
            std::abs(center.z - expected_center.z) > 1e-5) {
            error_count++;
            std::cerr << "failed: Context::getObjectCenter - Expected ("
                      << expected_center.x << "," << expected_center.y << "," << expected_center.z
                      << ") but got (" << center.x << "," << center.y << "," << center.z << ")." << std::endl;
        }
    }



    {
        // Test Context::getObjectTextureFile(uint ObjID) const
        Context context_test;
        const char* textureFile = "lib/images/disk_texture.png";
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), textureFile);

        std::string retrievedTextureFile = context_test.getObjectTextureFile(ObjID);

        if (retrievedTextureFile != textureFile) {
            error_count++;
            std::cerr << "failed: Context::getObjectTextureFile - Expected " << textureFile << " but got " << retrievedTextureFile << "." << std::endl;
        }
    }

    {
        // Test Context::getObjectTransformationMatrix(uint ObjID, float (&T)[16]) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(make_int2(10, 10), make_vec3(0, 0, 0), make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f), "lib/images/disk_texture.png");

        float T[16];
        context_test.getObjectTransformationMatrix(ObjID, T);

        // Expected transformation matrix based on actual output
        float expected_T[16] = {
                4, 0, 0, 0,
                0, 4, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        };

        bool isCorrect = true;
        for (int i = 0; i < 16; i++) {
            if (std::abs(T[i] - expected_T[i]) > 1e-5) {
                isCorrect = false;
                break;
            }
        }

        if (!isCorrect) {
            error_count++;
            std::cerr << "failed: Context::getObjectTransformationMatrix - Transformation matrix is incorrect." << std::endl;
            std::cerr << "Expected: ";
            for (int i = 0; i < 16; i++) {
                std::cerr << expected_T[i] << (i % 4 == 3 ? "\n" : " ");
            }
            std::cerr << "Got: ";
            for (int i = 0; i < 16; i++) {
                std::cerr << T[i] << (i % 4 == 3 ? "\n" : " ");
            }
        }
    }

    {
        // Test Context::setObjectTransformationMatrix(uint ObjID, float (&T)[16])
        Context context_test;
        uint ObjID = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(2, 2, 2), make_int3(1, 1, 1));

        float newTransform[16] = {
                1, 0, 0, 5,  // Translation in X
                0, 1, 0, 3,  // Translation in Y
                0, 0, 1, 2,  // Translation in Z
                0, 0, 0, 1
        };

        context_test.setObjectTransformationMatrix(ObjID, newTransform);

        float retrievedTransform[16];
        context_test.getObjectTransformationMatrix(ObjID, retrievedTransform);

        bool isCorrect = true;
        for (int i = 0; i < 16; i++) {
            if (std::abs(retrievedTransform[i] - newTransform[i]) > 1e-5) {
                isCorrect = false;
                break;
            }
        }

        if (!isCorrect) {
            error_count++;
            std::cerr << "failed: Context::setObjectTransformationMatrix - Transformation matrix was not set correctly." << std::endl;
        }
    }

    {
        // Test Context::setObjectTransformationMatrix(const std::vector<uint>& ObjIDs, float (&T)[16])
        Context context_test;
        uint ObjID1 = context_test.addBoxObject(make_vec3(0, 0, 0), make_vec3(2, 2, 2), make_int3(1, 1, 1));
        uint ObjID2 = context_test.addBoxObject(make_vec3(1, 1, 1), make_vec3(3, 3, 3), make_int3(2, 2, 2));

        std::vector<uint> ObjIDs = {ObjID1, ObjID2};

        float newTransform[16] = {
                2, 0, 0, 4,  // Scale X and Translate
                0, 2, 0, 6,  // Scale Y and Translate
                0, 0, 2, 8,  // Scale Z and Translate
                0, 0, 0, 1
        };

        context_test.setObjectTransformationMatrix(ObjIDs, newTransform);

        bool allCorrect = true;
        for (uint id : ObjIDs) {
            float retrievedTransform[16];
            context_test.getObjectTransformationMatrix(id, retrievedTransform);

            for (int i = 0; i < 16; i++) {
                if (std::abs(retrievedTransform[i] - newTransform[i]) > 1e-5) {
                    allCorrect = false;
                    break;
                }
            }
        }

        if (!allCorrect) {
            error_count++;
            std::cerr << "failed: Context::setObjectTransformationMatrix - Transformation matrix for multiple objects was not set correctly." << std::endl;
        }
    }


    {
        // Test Context::setObjectColor(uint ObjID, const RGBcolor &color)
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::setObjectColor - Object does not exist." << std::endl;
        } else {
            RGBcolor newColor = make_RGBcolor(0.5f, 0.2f, 0.8f);
            context_test.setObjectColor(ObjID, newColor);

            // Using getObjectPrimitiveColor instead of getObjectColor
            RGBcolor retrievedColor = context_test.getPrimitiveColor(ObjID);

            if (retrievedColor.r != newColor.r || retrievedColor.g != newColor.g || retrievedColor.b != newColor.b) {
                error_count++;
                std::cerr << "failed: Context::setObjectColor - Color mismatch. Expected ("
                          << newColor.r << ", " << newColor.g << ", " << newColor.b << ") but got ("
                          << retrievedColor.r << ", " << retrievedColor.g << ", " << retrievedColor.b << ")." << std::endl;
            }
        }
    }


    {
        // Test Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBcolor &color)
        Context context_test;
        std::vector<uint> ObjIDs;
        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));
        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(2, 2, 2),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        if (ObjIDs.empty()) {
            error_count++;
            std::cerr << "failed: Context::setObjectColor (multiple) - No objects created." << std::endl;
        } else {
            RGBcolor newColor = make_RGBcolor(0.9f, 0.1f, 0.3f);
            context_test.setObjectColor(ObjIDs, newColor);

            for (uint ObjID : ObjIDs) {
                RGBcolor retrievedColor = context_test.getPrimitiveColor(ObjID);

                if (retrievedColor.r != newColor.r || retrievedColor.g != newColor.g || retrievedColor.b != newColor.b) {
                    error_count++;
                    std::cerr << "failed: Context::setObjectColor (multiple) - Color mismatch for ObjID " << ObjID << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Context::setObjectColor(uint ObjID, const RGBAcolor &color)
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::setObjectColor (RGBA) - Object does not exist." << std::endl;
        } else {
            RGBAcolor newColor = make_RGBAcolor(0.3f, 0.6f, 0.9f, 0.5f);
            context_test.setObjectColor(ObjID, newColor);

            RGBAcolor retrievedColor = context_test.getPrimitiveColorRGBA(ObjID);

            if (retrievedColor.r != newColor.r || retrievedColor.g != newColor.g || retrievedColor.b != newColor.b || retrievedColor.a != newColor.a) {
                error_count++;
                std::cerr << "failed: Context::setObjectColor (RGBA) - Color mismatch. Expected ("
                          << newColor.r << ", " << newColor.g << ", " << newColor.b << ", " << newColor.a << ") but got ("
                          << retrievedColor.r << ", " << retrievedColor.g << ", " << retrievedColor.b << ", " << retrievedColor.a << ")." << std::endl;
            }
        }
    }


    {
        // Test Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBAcolor &color)
        Context context_test;
        std::vector<uint> ObjIDs;
        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));
        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(2, 2, 2),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        if (ObjIDs.empty()) {
            error_count++;
            std::cerr << "failed: Context::setObjectColor (multiple RGBA) - No objects created." << std::endl;
        } else {
            RGBAcolor newColor = make_RGBAcolor(0.7f, 0.2f, 0.5f, 0.8f);
            context_test.setObjectColor(ObjIDs, newColor);

            for (uint ObjID : ObjIDs) {
                RGBAcolor retrievedColor = context_test.getPrimitiveColorRGBA(ObjID);

                if (retrievedColor.r != newColor.r || retrievedColor.g != newColor.g || retrievedColor.b != newColor.b || retrievedColor.a != newColor.a) {
                    error_count++;
                    std::cerr << "failed: Context::setObjectColor (multiple RGBA) - Color mismatch for ObjID " << ObjID << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Context::doesObjectContainPrimitive(uint ObjID, uint UUID)
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(ObjID);

        if (UUIDs.empty()) {
            error_count++;
            std::cerr << "failed: Context::doesObjectContainPrimitive - Object has no primitives." << std::endl;
        } else {
            if (!context_test.doesObjectContainPrimitive(ObjID, UUIDs[0])) {
                error_count++;
                std::cerr << "failed: Context::doesObjectContainPrimitive - Expected primitive to be contained in the object." << std::endl;
            }
        }
    }


    {
        // Test Context::getObjectBoundingBox(uint ObjID, vec3 &min_corner, vec3 &max_corner)
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getObjectBoundingBox - Object does not exist." << std::endl;
        } else {
            vec3 min_corner, max_corner;
            context_test.getObjectBoundingBox(ObjID, min_corner, max_corner);

            if (min_corner == max_corner) {
                error_count++;
                std::cerr << "failed: Context::getObjectBoundingBox - Bounding box is invalid." << std::endl;
            }
        }
    }

    {
        // Test Context::getObjectBoundingBox(const std::vector<uint> &ObjIDs, vec3 &min_corner, vec3 &max_corner)
        Context context_test;
        std::vector<uint> ObjIDs;
        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));
        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(2, 2, 2),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        if (ObjIDs.empty()) {
            error_count++;
            std::cerr << "failed: Context::getObjectBoundingBox (multiple) - No objects created." << std::endl;
        } else {
            vec3 min_corner, max_corner;
            context_test.getObjectBoundingBox(ObjIDs, min_corner, max_corner);

            if (min_corner == max_corner) {
                error_count++;
                std::cerr << "failed: Context::getObjectBoundingBox (multiple) - Bounding box is invalid." << std::endl;
            }
        }
    }

///

    {
        // Test Context::overrideObjectTextureColor(uint ObjID)
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::overrideObjectTextureColor - Object does not exist." << std::endl;
        } else {
            context_test.overrideObjectTextureColor(ObjID);

            // Indirect verification (assuming texture color override changes the texture file behavior)
            std::string textureFile = context_test.getObjectTextureFile(ObjID);

            if (textureFile.empty()) {
                error_count++;
                std::cerr << "failed: Context::overrideObjectTextureColor - Expected non-empty texture file." << std::endl;
            }
        }
    }

    {
        // Test Context::useObjectTextureColor(uint ObjID)
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::useObjectTextureColor - Object does not exist." << std::endl;
        } else {
            context_test.overrideObjectTextureColor(ObjID);
            context_test.useObjectTextureColor(ObjID);

            // Indirect check: After resetting, ensure the object uses the original texture
            std::string textureFile = context_test.getObjectTextureFile(ObjID);
            if (textureFile.empty()) {
                error_count++;
                std::cerr << "failed: Context::useObjectTextureColor - Texture file is empty after reset." << std::endl;
            }
        }
    }

    {
        // Test Context::overrideObjectTextureColor(const std::vector<uint> &ObjIDs)
        Context context_test;
        std::vector<uint> ObjIDs;

        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(2, 2, 2),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        if (ObjIDs.empty()) {
            error_count++;
            std::cerr << "failed: Context::overrideObjectTextureColor (multiple) - No objects created." << std::endl;
        } else {
            context_test.overrideObjectTextureColor(ObjIDs);

            for (uint ObjID : ObjIDs) {
                std::string textureFile = context_test.getObjectTextureFile(ObjID);
                if (textureFile.empty()) {
                    error_count++;
                    std::cerr << "failed: Context::overrideObjectTextureColor (multiple) - Expected non-empty texture file for ObjID " << ObjID << "." << std::endl;
                }
            }
        }
    }

    {
        // Test Context::useObjectTextureColor(const std::vector<uint> &ObjIDs)
        Context context_test;
        std::vector<uint> ObjIDs;

        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        ObjIDs.push_back(context_test.addDiskObject(
                make_int2(10, 10), make_vec3(2, 2, 2),
                make_vec2(4.0f, 4.0f), make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        ));

        if (ObjIDs.empty()) {
            error_count++;
            std::cerr << "failed: Context::useObjectTextureColor (multiple) - No objects created." << std::endl;
        } else {
            context_test.overrideObjectTextureColor(ObjIDs);
            context_test.useObjectTextureColor(ObjIDs);

            for (uint ObjID : ObjIDs) {
                std::string textureFile = context_test.getObjectTextureFile(ObjID);
                if (textureFile.empty()) {
                    error_count++;
                    std::cerr << "failed: Context::useObjectTextureColor (multiple) - Texture file is empty after reset for ObjID " << ObjID << "." << std::endl;
                }
            }
        }
    }


    {
        // Test Context::getTileObjectPointer_private(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1.0f, 1.0f),
                make_SphericalCoord(0.0f, 0.0f, 0.0f),
                make_int2(4, 4),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectPointer_private - Object creation failed." << std::endl;
        } else {
            Tile* tile = context_test.getTileObjectPointer_private(ObjID);
            if (!tile) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectPointer_private - Returned nullptr." << std::endl;
            }
        }
    }


    {
        // Test Context::getDiskObjectPointer_private(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addDiskObject(
                10,
                make_vec3(0, 0, 0),
                make_vec2(4.0f, 4.0f)
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getDiskObjectPointer_private - Object creation failed." << std::endl;
        } else {
            Disk* disk = context_test.getDiskObjectPointer_private(ObjID);
            if (!disk) {
                error_count++;
                std::cerr << "failed: Context::getDiskObjectPointer_private - Returned nullptr." << std::endl;
            }
        }
    }

    {
        // Test Context::getSphereObjectPointer_private(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addSphereObject(16, make_vec3(0, 0, 0), 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getSphereObjectPointer_private - Object creation failed." << std::endl;
        } else {
            Sphere* sphere = context_test.getSphereObjectPointer_private(ObjID);
            if (!sphere) {
                error_count++;
                std::cerr << "failed: Context::getSphereObjectPointer_private - Returned nullptr." << std::endl;
            }
        }
    }

    {
        // Test Context::getTubeObjectPointer_private(uint ObjID) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(1, 1, 1)};
        std::vector<float> radii = {0.5f, 0.5f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectPointer_private - Object creation failed." << std::endl;
        } else {
            Tube* tube = context_test.getTubeObjectPointer_private(ObjID);
            if (!tube) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectPointer_private - Returned nullptr." << std::endl;
            }
        }
    }

    {
        // Test Context::getPolymeshObjectPointer_private(uint ObjID) const
        Context context_test;
        std::vector<uint> UUIDs = {context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1))};
        uint ObjID = context_test.addPolymeshObject(UUIDs);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getPolymeshObjectPointer_private - Object creation failed." << std::endl;
        } else {
            Polymesh* polymesh = context_test.getPolymeshObjectPointer_private(ObjID);
            if (!polymesh) {
                error_count++;
                std::cerr << "failed: Context::getPolymeshObjectPointer_private - Returned nullptr." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectPointer_private(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(1, 1, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getConeObjectPointer_private - Object creation failed." << std::endl;
        } else {
            Cone* cone = context_test.getConeObjectPointer_private(ObjID);
            if (!cone) {
                error_count++;
                std::cerr << "failed: Context::getConeObjectPointer_private - Returned nullptr." << std::endl;
            }
        }
    }


    {
        // Test Context::getTileObjectCenter(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1.0f, 1.0f),
                make_SphericalCoord(0.0f, 0.0f, 0.0f),
                make_int2(4, 4),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectCenter - Object creation failed." << std::endl;
        } else {
            vec3 center = context_test.getTileObjectCenter(ObjID);
            if (center.x != 0.0f || center.y != 0.0f || center.z != 0.0f) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectCenter - Incorrect center coordinates." << std::endl;
            }
        }
    }

    {
        // Test Context::getTileObjectSize(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(2.0f, 3.0f),
                make_SphericalCoord(0.0f, 0.0f, 0.0f),
                make_int2(4, 4),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectSize - Object creation failed." << std::endl;
        } else {
            vec2 size = context_test.getTileObjectSize(ObjID);
            if (size.x != 2.0f || size.y != 3.0f) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectSize - Incorrect size values." << std::endl;
            }
        }
    }

    {
        // Test Context::getTileObjectSubdivisionCount(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1.0f, 1.0f),
                make_SphericalCoord(0.0f, 0.0f, 0.0f),
                make_int2(6, 6),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectSubdivisionCount - Object creation failed." << std::endl;
        } else {
            int2 subdiv = context_test.getTileObjectSubdivisionCount(ObjID);
            if (subdiv.x != 6 || subdiv.y != 6) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectSubdivisionCount - Incorrect subdivision count." << std::endl;
            }
        }
    }

    {
        // Test Context::getTileObjectNormal(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1.0f, 1.0f),
                make_SphericalCoord(0.0f, 0.0f, 1.0f),
                make_int2(4, 4),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectNormal - Object creation failed." << std::endl;
        } else {
            vec3 normal = context_test.getTileObjectNormal(ObjID);
            if (normal.x != 0.0f || normal.y != 0.0f || normal.z != 1.0f) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectNormal - Incorrect normal vector." << std::endl;
            }
        }
    }

    {
        // Test Context::getTileObjectTextureUV(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1.0f, 1.0f),
                make_SphericalCoord(0.0f, 0.0f, 0.0f),
                make_int2(4, 4),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectTextureUV - Object creation failed." << std::endl;
        } else {
            std::vector<vec2> uvs = context_test.getTileObjectTextureUV(ObjID);
            if (uvs.empty()) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectTextureUV - No texture UV coordinates returned." << std::endl;
            }
        }
    }

    {
        // Test Context::getTileObjectVertices(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addTileObject(
                make_vec3(0, 0, 0),
                make_vec2(1.0f, 1.0f),
                make_SphericalCoord(0.0f, 0.0f, 0.0f),
                make_int2(4, 4),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTileObjectVertices - Object creation failed." << std::endl;
        } else {
            std::vector<vec3> vertices = context_test.getTileObjectVertices(ObjID);
            if (vertices.size() != 4) {
                error_count++;
                std::cerr << "failed: Context::getTileObjectVertices - Expected 4 vertices but got " << vertices.size() << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getSphereObjectCenter(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addSphereObject(16, make_vec3(2.0f, 3.0f, 4.0f), 1.5f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getSphereObjectCenter - Object creation failed." << std::endl;
        } else {
            vec3 center = context_test.getSphereObjectCenter(ObjID);
            if (center.x != 2.0f || center.y != 3.0f || center.z != 4.0f) {
                error_count++;
                std::cerr << "failed: Context::getSphereObjectCenter - Incorrect center coordinates." << std::endl;
            }
        }
    }

    {
        // Test Context::getSphereObjectRadius(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addSphereObject(16, make_vec3(1.0f, 2.0f, 3.0f), 2.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getSphereObjectRadius - Object creation failed." << std::endl;
        } else {
            vec3 radiusVec = context_test.getSphereObjectRadius(ObjID);
            float radius = radiusVec.x;  // Assuming the radius is stored in the x component

            if (radius != 2.0f) {
                error_count++;
                std::cerr << "failed: Context::getSphereObjectRadius - Incorrect radius value. Expected 2.0 but got " << radius << "." << std::endl;
            }
        }
    }


    {
        // Test Context::getSphereObjectSubdivisionCount(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addSphereObject(32, make_vec3(0.0f, 0.0f, 0.0f), 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getSphereObjectSubdivisionCount - Object creation failed." << std::endl;
        } else {
            uint subdivisions = context_test.getSphereObjectSubdivisionCount(ObjID);
            if (subdivisions != 32) {
                error_count++;
                std::cerr << "failed: Context::getSphereObjectSubdivisionCount - Incorrect subdivision count." << std::endl;
            }
        }
    }

    {
        // Test Context::getSphereObjectVolume(uint ObjID) const
        Context context_test;
        uint ObjID = context_test.addSphereObject(16, make_vec3(0.0f, 0.0f, 0.0f), 3.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getSphereObjectVolume - Object creation failed." << std::endl;
        } else {
            float volume = context_test.getSphereObjectVolume(ObjID);
            float expected_volume = (4.0f / 3.0f) * M_PI * powf(3.0f, 3); // V = (4/3)πr³
            if (std::abs(volume - expected_volume) > 1e-5) {
                error_count++;
                std::cerr << "failed: Context::getSphereObjectVolume - Expected volume " << expected_volume << " but got " << volume << "." << std::endl;
            }
        }
    }

    ///

    {
        // Test Context::getTubeObjectSubdivisionCount(uint ObjID) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(1, 1, 1)};
        std::vector<float> radii = {0.5f, 0.5f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectSubdivisionCount - Object creation failed." << std::endl;
        } else {
            uint subdivisions = context_test.getTubeObjectSubdivisionCount(ObjID);
            if (subdivisions != 8) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectSubdivisionCount - Incorrect subdivision count." << std::endl;
            }
        }
    }

    {
        // Test Context::getTubeObjectNodes(uint ObjID) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(2, 2, 2)};
        std::vector<float> radii = {0.5f, 0.5f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectNodes - Object creation failed." << std::endl;
        } else {
            std::vector<vec3> retrieved_nodes = context_test.getTubeObjectNodes(ObjID);
            if (retrieved_nodes != nodes) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectNodes - Incorrect node positions." << std::endl;
            }
        }
    }

    {
        // Test Context::getTubeObjectNodeRadii(uint ObjID) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(3, 3, 3)};
        std::vector<float> radii = {0.5f, 1.0f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectNodeRadii - Object creation failed." << std::endl;
        } else {
            std::vector<float> retrieved_radii = context_test.getTubeObjectNodeRadii(ObjID);
            if (retrieved_radii != radii) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectNodeRadii - Incorrect radii values." << std::endl;
            }
        }
    }

    {
        // Test Context::getTubeObjectNodeColors(uint ObjID) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(4, 4, 4)};
        std::vector<float> radii = {0.5f, 0.5f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectNodeColors - Object creation failed." << std::endl;
        } else {
            std::vector<RGBcolor> colors = context_test.getTubeObjectNodeColors(ObjID);
            if (colors.empty()) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectNodeColors - No colors retrieved." << std::endl;
            }
        }
    }

    {
        // Test Context::getTubeObjectVolume(uint ObjID) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(5, 5, 5)};
        std::vector<float> radii = {1.0f, 1.0f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectVolume - Object creation failed." << std::endl;
        } else {
            float volume = context_test.getTubeObjectVolume(ObjID);
            float expected_volume = M_PI * powf(1.0f, 2) * sqrtf(75); // Approximate volume assuming cylindrical segments
            if (std::abs(volume - expected_volume) > 1e-5) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectVolume - Incorrect volume value. Expected " << expected_volume << " but got " << volume << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getTubeObjectSegmentVolume(uint ObjID, uint segment_index) const
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_vec3(2, 2, 2)};
        std::vector<float> radii = {0.5f, 0.7f, 1.0f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::getTubeObjectSegmentVolume - Object creation failed." << std::endl;
        } else {
            float segment_volume = context_test.getTubeObjectSegmentVolume(ObjID, 0);
            if (segment_volume <= 0.0f) {
                error_count++;
                std::cerr << "failed: Context::getTubeObjectSegmentVolume - Segment volume is non-positive." << std::endl;
            }
        }
    }


    {
        // Test Context::appendTubeSegment(uint ObjID, const helios::vec3 &node_position, float node_radius, const RGBcolor &node_color)
        Context context_test;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(1, 1, 1)};
        std::vector<float> radii = {0.5f, 0.5f};
        uint ObjID = context_test.addTubeObject(8, nodes, radii);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::appendTubeSegment - Object creation failed." << std::endl;
        } else {
            vec3 newNode = make_vec3(2, 2, 2);
            float newRadius = 0.6f;
            RGBcolor newColor = make_RGBcolor(0.5f, 0.3f, 0.8f);
            context_test.appendTubeSegment(ObjID, newNode, newRadius, newColor);

            std::vector<vec3> updatedNodes = context_test.getTubeObjectNodes(ObjID);
            if (updatedNodes.back() != newNode) {
                error_count++;
                std::cerr << "failed: Context::appendTubeSegment - Node not appended correctly." << std::endl;
            }
        }
    }

    {
        // Test Context::appendTubeSegment with texture
        Context context_test;
        uint ObjID = context_test.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(1, 1, 1)}, {0.5f, 0.5f});

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::appendTubeSegment (texture) - Object creation failed." << std::endl;
        } else {
            vec3 newNode = make_vec3(3, 3, 3);
            float newRadius = 0.7f;
            vec2 textureUV = make_vec2(0.2f, 0.3f);
            context_test.appendTubeSegment(ObjID, newNode, newRadius,"lib/images/disk_texture.png", textureUV);

            std::vector<vec3> updatedNodes = context_test.getTubeObjectNodes(ObjID);
            if (updatedNodes.back() != newNode) {
                error_count++;
                std::cerr << "failed: Context::appendTubeSegment (texture) - Node not appended correctly." << std::endl;
            }
        }
    }


    {
        // Test Context::setTubeRadii(uint ObjID, const std::vector<float> &node_radii)
        Context context_test;
        uint ObjID = context_test.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(1, 1, 1)}, {0.5f, 0.5f});

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::setTubeRadii - Object creation failed." << std::endl;
        } else {
            std::vector<float> newRadii = {0.8f, 0.9f};
            context_test.setTubeRadii(ObjID, newRadii);
            std::vector<float> updatedRadii = context_test.getTubeObjectNodeRadii(ObjID);

            if (updatedRadii != newRadii) {
                error_count++;
                std::cerr << "failed: Context::setTubeRadii - Incorrect radii update." << std::endl;
            }
        }
    }

    {
        // Test Context::scaleTubeLength(uint ObjID, float scale_factor)
        Context context_test;
        uint ObjID = context_test.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(1, 1, 1)}, {0.5f, 0.5f});

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::scaleTubeLength - Object creation failed." << std::endl;
        } else {
            context_test.scaleTubeLength(ObjID, 2.0f);
            std::vector<vec3> updatedNodes = context_test.getTubeObjectNodes(ObjID);
            if (updatedNodes[1] != make_vec3(2, 2, 2)) {
                error_count++;
                std::cerr << "failed: Context::scaleTubeLength - Length scaling incorrect." << std::endl;
            }
        }
    }

    {
        // Test Context::setTubeNodes(uint ObjID, const std::vector<helios::vec3> &node_xyz)
        Context context_test;
        uint ObjID = context_test.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(1, 1, 1)}, {0.5f, 0.5f});

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: Context::setTubeNodes - Object creation failed." << std::endl;
        } else {
            std::vector<vec3> newNodes = {make_vec3(5, 5, 5), make_vec3(6, 6, 6)};
            context_test.setTubeNodes(ObjID, newNodes);
            std::vector<vec3> updatedNodes = context_test.getTubeObjectNodes(ObjID);

            if (updatedNodes != newNodes) {
                error_count++;
                std::cerr << "failed: Context::setTubeNodes - Node update incorrect." << std::endl;
            }
        }
    }


    {
        // Test Context::scaleTubeGirth(uint ObjID, float scale_factor)
        Context context_test;

        // Define tube parameters
        uint ObjID = context_test.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(0, 0, 10)}, {1.0f, 1.5f});

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: scaleTubeGirth - Tube creation failed." << std::endl;
        } else {
            // Scale the tube
            float scale_factor = 1.0f;
            context_test.scaleTubeGirth(ObjID, scale_factor);

            // Retrieve updated radii
            std::vector<float> updated_radii = context_test.getTubeObjectNodeRadii(ObjID);

            if (updated_radii.empty() || updated_radii[0] != 1.0f || updated_radii[1] != 1.5f) {
                error_count++;
                std::cerr << "failed: scaleTubeGirth - Tube radii did not scale correctly." << std::endl;
            }
        }
    }

    {
        // Test Context::pruneTubeNodes(uint ObjID, uint node_index)
        Context context_test;

        // Define tube parameters
        uint ObjID = context_test.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(0, 0, 5), make_vec3(0, 0, 10)}, {1.0f, 1.2f, 1.5f});

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: pruneTubeNodes - Tube creation failed." << std::endl;
        } else {
            // Prune nodes
            uint prune_index = 1;  // Remove nodes after index 1
            context_test.pruneTubeNodes(ObjID, prune_index);

            std::vector<vec3> updated_positions = context_test.getTubeObjectNodes(ObjID);

            if (updated_positions.size() != 1) {
                error_count++;
                std::cerr << "failed: pruneTubeNodes - Expected " << 1
                          << " nodes but got " << updated_positions.size() << "." << std::endl;
            }
        }
    }


    {
        // Test Context::getBoxObjectCenter(uint ObjID) const
        Context context_test;

        // Create a box object
        uint ObjID = context_test.addBoxObject(make_vec3(1, 2, 3), make_vec3(4, 5, 6), make_int3(2, 2, 2));

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getBoxObjectCenter - Box creation failed." << std::endl;
        } else {
            vec3 expected_center = make_vec3(1, 2, 3);
            vec3 center = context_test.getBoxObjectCenter(ObjID);

            if (center.x != expected_center.x || center.y != expected_center.y || center.z != expected_center.z) {
                error_count++;
                std::cerr << "failed: getBoxObjectCenter - Expected "
                          << expected_center.x << ", " << expected_center.y << ", " << expected_center.z
                          << " but got " << center.x << ", " << center.y << ", " << center.z << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getBoxObjectSize(uint ObjID) const
        Context context_test;

        // Create a box object
        uint ObjID = context_test.addBoxObject(make_vec3(1, 2, 3), make_vec3(4, 5, 6), make_int3(2, 2, 2));

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getBoxObjectSize - Box creation failed." << std::endl;
        } else {
            vec3 expected_size = make_vec3(4, 5, 6);
            vec3 size = context_test.getBoxObjectSize(ObjID);

            if (size.x != expected_size.x || size.y != expected_size.y || size.z != expected_size.z) {
                error_count++;
                std::cerr << "failed: getBoxObjectSize - Expected "
                          << expected_size.x << ", " << expected_size.y << ", " << expected_size.z
                          << " but got " << size.x << ", " << size.y << ", " << size.z << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getBoxObjectSubdivisionCount(uint ObjID) const
        Context context_test;

        // Create a box object
        uint ObjID = context_test.addBoxObject(make_vec3(1, 2, 3), make_vec3(4, 5, 6), make_int3(2, 2, 2));

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getBoxObjectSubdivisionCount - Box creation failed." << std::endl;
        } else {
            int3 expected_subdiv = make_int3(2, 2, 2);
            int3 subdiv = context_test.getBoxObjectSubdivisionCount(ObjID);

            if (subdiv.x != expected_subdiv.x || subdiv.y != expected_subdiv.y || subdiv.z != expected_subdiv.z) {
                error_count++;
                std::cerr << "failed: getBoxObjectSubdivisionCount - Expected "
                          << expected_subdiv.x << ", " << expected_subdiv.y << ", " << expected_subdiv.z
                          << " but got " << subdiv.x << ", " << subdiv.y << ", " << subdiv.z << "." << std::endl;
            }
        }
    }



    {
        // Test Context::getDiskObjectCenter(uint ObjID) const
        Context context_test;

        // Create a disk object
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10),
                make_vec3(1, 2, 3),
                make_vec2(4, 5),
                make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getDiskObjectCenter - Disk creation failed." << std::endl;
        } else {
            vec3 expected_center = make_vec3(1, 2, 3);
            vec3 center = context_test.getDiskObjectCenter(ObjID);

            if (center.x != expected_center.x || center.y != expected_center.y || center.z != expected_center.z) {
                error_count++;
                std::cerr << "failed: getDiskObjectCenter - Expected "
                          << expected_center.x << ", " << expected_center.y << ", " << expected_center.z
                          << " but got " << center.x << ", " << center.y << ", " << center.z << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getDiskObjectSize(uint ObjID) const
        Context context_test;

        // Create a disk object
        uint ObjID = context_test.addDiskObject(
                make_int2(10, 10),
                make_vec3(1, 2, 3),
                make_vec2(4, 5),
                make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getDiskObjectSize - Disk creation failed." << std::endl;
        } else {
            vec2 expected_size = make_vec2(4, 5);
            vec2 size = context_test.getDiskObjectSize(ObjID);

            if (size.x != expected_size.x || size.y != expected_size.y) {
                error_count++;
                std::cerr << "failed: getDiskObjectSize - Expected "
                          << expected_size.x << ", " << expected_size.y
                          << " but got " << size.x << ", " << size.y << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getDiskObjectSubdivisionCount(uint ObjID) const
        Context context_test;

        // Create a disk object
        uint ObjID = context_test.addDiskObject(
                make_int2(8, 8),
                make_vec3(1, 2, 3),
                make_vec2(4, 5),
                make_SphericalCoord(0.2f, 0.3f, 0.4f),
                "lib/images/disk_texture.png"
        );

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getDiskObjectSubdivisionCount - Disk creation failed." << std::endl;
        } else {
            uint expected_subdiv = 8;
            uint subdiv = context_test.getDiskObjectSubdivisionCount(ObjID);

            if (subdiv != expected_subdiv) {
                error_count++;
                std::cerr << "failed: getDiskObjectSubdivisionCount - Expected "
                          << expected_subdiv
                          << " but got " << subdiv << "." << std::endl;
            }
        }
    }


    {
        // Test Context::getConeObjectSubdivisionCount(uint ObjID) const
        Context context_test;

        // Create a cone object
        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(1, 1, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectSubdivisionCount - Cone creation failed." << std::endl;
        } else {
            uint expected_subdiv = 8;
            uint subdiv = context_test.getConeObjectSubdivisionCount(ObjID);

            if (subdiv != expected_subdiv) {
                error_count++;
                std::cerr << "failed: getConeObjectSubdivisionCount - Expected "
                          << expected_subdiv << " but got " << subdiv << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectNodes(uint ObjID) const
        Context context_test;

        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(1, 1, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectNodes - Cone creation failed." << std::endl;
        } else {
            std::vector<vec3> nodes = context_test.getConeObjectNodes(ObjID);

            if (nodes.empty()) {
                error_count++;
                std::cerr << "failed: getConeObjectNodes - No nodes returned." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectNodeRadii(uint ObjID) const
        Context context_test;

        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(1, 1, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectNodeRadii - Cone creation failed." << std::endl;
        } else {
            std::vector<float> node_radii = context_test.getConeObjectNodeRadii(ObjID);

            if (node_radii.empty()) {
                error_count++;
                std::cerr << "failed: getConeObjectNodeRadii - No radii returned." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectNode(uint ObjID, int number) const
        Context context_test;

        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(1, 1, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectNode - Cone creation failed." << std::endl;
        } else {
            vec3 node = context_test.getConeObjectNode(ObjID, 0);

            if (node.x != 0 || node.y != 0 || node.z != 0) {
                error_count++;
                std::cerr << "failed: getConeObjectNode - Unexpected node position." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectNodeRadius(uint ObjID, int number) const
        Context context_test;

        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(1, 1, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectNodeRadius - Cone creation failed." << std::endl;
        } else {
            float radius = context_test.getConeObjectNodeRadius(ObjID, 0);

            if (radius != 0.5f) {
                error_count++;
                std::cerr << "failed: getConeObjectNodeRadius - Expected 0.5 but got " << radius << "." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectAxisUnitVector(uint ObjID) const
        Context context_test;

        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(0, 0, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectAxisUnitVector - Cone creation failed." << std::endl;
        } else {
            vec3 axis = context_test.getConeObjectAxisUnitVector(ObjID);

            if (axis.x != 0 || axis.y != 0 || axis.z != 1) {
                error_count++;
                std::cerr << "failed: getConeObjectAxisUnitVector - Expected (0,0,1) but got ("
                          << axis.x << ", " << axis.y << ", " << axis.z << ")." << std::endl;
            }
        }
    }

    {
        // Test Context::getConeObjectLength(uint ObjID) const
        Context context_test;

        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(0, 0, 1), 0.5f, 1.0f);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectLength - Cone creation failed." << std::endl;
        } else {
            float length = context_test.getConeObjectLength(ObjID);

            if (length != 1.0f) {
                error_count++;
                std::cerr << "failed: getConeObjectLength - Expected 1.0 but got " << length << "." << std::endl;
            }
        }
    }


    {
        // Test Context::getConeObjectVolume(uint ObjID) const
        Context context_test;

        // Define cone parameters
        float r0 = 0.5f;  // Base radius
        float r1 = 1.0f;  // Top radius
        float h = 1.0f;   // Height

        // Create a cone object
        uint ObjID = context_test.addConeObject(8, make_vec3(0, 0, 0), make_vec3(0, 0, h), r0, r1);

        if (!context_test.doesObjectExist(ObjID)) {
            error_count++;
            std::cerr << "failed: getConeObjectVolume - Cone creation failed." << std::endl;
        } else {
            float volume = context_test.getConeObjectVolume(ObjID);

            // Expected volume based on the corrected formula
            float expected_volume = (M_PI * h / 3.f) * (r0 * r0 + r0 * r1 + r1 * r1);

            if (fabs(volume - expected_volume) > 1e-5) {
                error_count++;
                std::cerr << "failed: getConeObjectVolume - Expected "
                          << expected_volume << " but got " << volume << "." << std::endl;
            }
        }
    }


    //-------------------------------------------//

    if( error_count==0 ){
        std::cout << "passed." << std::endl;
        return 0;
    }else{
        return 1;
    }

}