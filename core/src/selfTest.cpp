/** \file "selfTest.cpp" Context selfTest() function.

    Copyright (C) 2016-2024 Brian Bailey

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
    if( fabsf( area_r - size.x*size.y )>1e-5 ){
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

    context_test.setPrimitiveData( UUID, "somedata", HELIOS_TYPE_FLOAT, cpdata.size(), &cpdata[0] );

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

    if( fabsf(center_cpy.x-center.x-shift.x)>errtol || fabsf(center_cpy.y-center.y-shift.y)>errtol || fabsf(center_cpy.z-center.z-shift.z)>errtol || center_r.x!=center.x || center_r.y!=center.y || center_r.z!=center.z ){
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
    rotation.elevation = 0.15f*M_PI;
    rotation.azimuth = 0.5f*M_PI;;
    UUID = context_test.addPatch(center,size,rotation);
    normal_r = context_test.getPrimitiveNormal(UUID);

    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    context_test.deletePrimitive(UUID);

    if( fabsf(rotation_r.elevation-rotation.elevation)>errtol || fabsf(rotation_r.azimuth-rotation.azimuth)>errtol ){
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

    if( fabsf(rotation_r.zenith-0.f)>errtol || fabsf(rotation_r.azimuth-0.f)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
    }

    normal_r = context_test.getPrimitiveNormal( UUIDs.at(2));
    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    if( fabsf(rotation_r.zenith-0.f)>errtol || fabsf(rotation_r.azimuth-0.5f*float(M_PI))>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
    }

    size_r = context_test.getPatchSize( UUIDs.at(0) );

    if( fabsf(size_r.x-size3.x)>errtol || fabsf(size_r.y-size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
    }

    size_r = context_test.getPatchSize( UUIDs.at(2) );

    if( fabsf(size_r.x-size3.y)>errtol || fabsf(size_r.y-size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
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

        if( fabsf(rotation_r.zenith-rotation.zenith)>errtol || fabsf(rotation_r.azimuth-rotation.azimuth)>errtol ){
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

    float area_exact = 0.25f*float(M_PI)*size.x*size.y;

    if( fabsf(At-area_exact)>0.005 ){
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

    if( fabsf( A_1 - scale*scale*A_0 )>1e-5 ){
        error_count ++;
        std::cerr << "failed: Patch scaling - scaled area not correct." << std::endl;
    }

    //------- Primitive Data --------//

    float data = 5;
    context_test.setPrimitiveData( UUID,"some_data",HELIOS_TYPE_FLOAT,1,&data);

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

    context_test.setPrimitiveData(UUID,"some_data",HELIOS_TYPE_FLOAT,1,&data);
    context_test.getPrimitiveData(UUID,"some_data",data_return);
    if( data_return!=data ){
        error_count ++;
        std::cerr << "failed: set/getPrimitiveData (setting/getting through Context). Get data did not match set data." << std::endl;
        std::cout << data_return << std::endl;
    }

    std::vector<float> data_v{0,1,2,3,4};

    context_test.setPrimitiveData( UUID,"some_data",HELIOS_TYPE_FLOAT,5,&data_v[0] );

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

    if( fabsf(Ap-0.25f*float(M_PI)*sizep.x*sizep.y)/(0.25f*float(M_PI)*sizep.x*sizep.y)>0.01f ){
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

    if( fabsf(area-sizep.x*sizep.y)>0.001f ){
        error_count ++;
        std::cerr << "failed: Patch masked with (u,v) coordinates did not return correct area." << std::endl;
    }

    uv0 = make_vec2( 0, 0 );
    uv1 = make_vec2( 1, 1 );
    uv2 = make_vec2( 0, 1 );

    uint UUIDp3 = context_test.addTriangle( make_vec3(2,3,4), make_vec3(2,3+sizep.y,4), make_vec3(2+sizep.x,3+sizep.y,4), texture, uv0, uv1, uv2 );

    area = context_test.getPrimitiveArea(UUIDp3);

    if( fabsf(area-0.5f*0.25f*float(M_PI)*sizep.x*sizep.y)>0.01f ){
        error_count ++;
        std::cerr << "failed: Triangle masked with (u,v) coordinates did not return correct area." << std::endl;
    }

    uint UUIDt2 = context_test.addTriangle( make_vec3(0,0,0), make_vec3(1,0,0), make_vec3(1,1,0), "lib/images/diamond_texture.png", make_vec2(0,0), make_vec2(1,0), make_vec2(1,1) );
    float solid_fraction = context_test.getPrimitiveSolidFraction(UUIDt2);
    if( fabs(solid_fraction-0.5f)>errtol ){
      error_count ++;
      std::cerr << "failed: Textured triangle solid fraction was not correct." << std::endl;
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

    std::vector<float> gdata_v{0,1,2,3,4};

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

    context_test.setObjectData(IDtile, "some_data",HELIOS_TYPE_FLOAT,5,&objdata_v[0] );

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
    std::vector<helios::vec3> nodes_T = context_test.getConeObjectPointer(cone_1)->getNodes();

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
    nodes_T = context_test.getConeObjectPointer(cone_1)->getNodes();

    if(nodes_T.at(0).x - 1.0f > errtol || nodes_T.at(0).y - 1.0f > errtol || nodes_T.at(0).z - 1.0f > errtol ||
       nodes_T.at(1).x - 3.0f > errtol || nodes_T.at(1).y - 1.0f > errtol || nodes_T.at(1).z - 1.0f > errtol ){
        error_count ++;
        std::cerr << "failed: rotate cone object. Node coordinates after rotation not correct." << std::endl;
    }


    //scale the length of the cone to twice its original length
    context_test.getConeObjectPointer(cone_1)->scaleLength(2.0);
    //get the updated node positions
    nodes_T = context_test.getConeObjectPointer(cone_1)->getNodes();

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

    if( fabsf(sph1.radius-sph2.radius)>errtol || fabsf(sph1.elevation-sph2.elevation)>errtol || fabsf(sph1.zenith-sph2.zenith)>errtol || fabsf(sph1.azimuth-sph2.azimuth)>errtol ){
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

    if( fabsf(xi.x-xi_ref.x)>errtol || fabsf(xi.y-xi_ref.y)>errtol || fabsf(xi.z-xi_ref.z)>errtol ){
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

    context_io.setGlobalData( "gdatad", HELIOS_TYPE_DOUBLE, gdatad_io.size(), &gdatad_io[0] );

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
    if(err_4 >= errtol){
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

    std::string ext = getFileExtension(filename);
    std::string name = getFileName(filename);
    std::string stem = getFileStem(filename);
    std::string path = getFilePath(filename,true);

    if( !ext.empty() || name!="filename" || stem!="filename" || path!="/path/to/.hidden/file/" ){
      std::cerr << "failed: file path parsing functions were not correct." << std::endl;
      error_count++;
    }

    filename = ".hidden/path/to/file/filename.ext";

    ext = getFileExtension(filename);
    name = getFileName(filename);
    stem = getFileStem(filename);
    path = getFilePath(filename,false);

    if( ext!=".ext" || name!="filename.ext" || stem!="filename" || path!=".hidden/path/to/file" ){
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

    //-------------------------------------------//

    if( error_count==0 ){
        std::cout << "passed." << std::endl;
        return 0;
    }else{
        return 1;
    }

}