/** \file "Context.cpp" Context declarations. 
    \author Brian Bailey

    Copyright (C) 2016-2021  Brian Bailey

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

Context::Context(){

    //---- ALL DEFAULT VALUES ARE SET HERE ----//

    iscontextinitialized=true;

    sim_date = make_Date(1,6,2000);

    sim_time = make_Time(12,0);

    scene_radius = 1.f;

    // --- Initialize random number generator ---- //

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

    // --- Set Geometry as `Clean' --- //

    isgeometrydirty = false;

    currentUUID = 0;

    currentObjectID = 1; //object ID of 0 is reserved for default object

}

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
    if( fabsf( area_r - size.x*size.y )>1e-5 ){
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

    std::vector<float> cpdata{5.2f,2.5f,3.1f};

    context_test.setPrimitiveData( UUID, "somedata", HELIOS_TYPE_FLOAT, cpdata.size(), &cpdata[0] );

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
    context_test.getPrimitivePointer(UUID_cpy)->translate(shift);
    center_cpy = context_test.getPatchPointer(UUID_cpy)->getCenter();
    center_r = context_test.getPatchPointer(UUID)->getCenter();

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
    prim = context_test.getPrimitivePointer(UUID);
    normal_r = prim->getNormal();

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
    objID = context_test.addBoxObject( center, size3, subdiv );
    UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();

    normal_r = context_test.getPrimitivePointer(UUIDs.at(0))->getNormal();
    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    if( fabsf(rotation_r.zenith-0.f)>errtol || fabsf(rotation_r.azimuth-0.f)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
    }

    normal_r = context_test.getPrimitivePointer(UUIDs.at(2))->getNormal();
    rotation_r = make_SphericalCoord( 0.5f*float(M_PI)-asinf( normal_r.z ), atan2f(normal_r.x,normal_r.y) );

    if( fabsf(rotation_r.zenith-0.f)>errtol || fabsf(rotation_r.azimuth-0.5f*M_PI)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face normals incorrect." << std::endl;
    }

    size_r = context_test.getPatchPointer(UUIDs.at(0))->getSize();

    if( fabsf(size_r.x-size3.x)>errtol || fabsf(size_r.y-size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
    }

    size_r = context_test.getPatchPointer(UUIDs.at(2))->getSize();

    if( fabsf(size_r.x-size3.y)>errtol || fabsf(size_r.y-size3.z)>errtol ){
        error_count++;
        std::cerr << "failed: addBox(). Face sizes incorrect." << std::endl;
    }

    //------- Add a Rotated Tile --------//

    center = make_vec3(1,2,3);
    size = make_vec2(3,2);
    int2 subdiv2(3,3);
    rotation = make_SphericalCoord( 0.25f*M_PI, 1.4f*M_PI );
    objID = context_test.addTileObject(center, size, subdiv2, rotation);
    UUIDs = context_test.getObjectPointer(objID)->getPrimitiveUUIDs();

    for( uint UUIDp : UUIDs){

        normal_r = context_test.getPrimitivePointer(UUIDp)->getNormal();

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

        float area = context_test.getPrimitivePointer(UUIDp)->getArea();

        At+=area;

    }

    float area_exact = 0.25f*float(M_PI)*size.x*size.y;

    if( fabsf(At-area_exact)>0.005 ){
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

    if( fabsf( A_1 - scale*scale*A_0 )>1e-5 ){
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

    std::vector<float> data_v{0,1,2,3,4};

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

    vec2 uv0(0,0);
    vec2 uv1(1,0);
    vec2 uv2(1,1);
    vec2 uv3(0,1);

    uint UUIDp = context_test.addPatch( make_vec3(2,3,4), sizep, make_SphericalCoord(0,0), texture, 0.5*(uv0+uv2), uv2-uv0 );

    if( !context_test.getPrimitivePointer(UUIDp)->hasTexture() ){
        error_count ++;
        std::cerr << "failed: Texture-mapped patch was found not to have texture." << std::endl;
    }

    std::string texture2 = context_test.getPrimitivePointer(UUIDp)->getTextureFile();

    if( texture2!=texture ){
        error_count ++;
        std::cerr << "failed: textures - queried texture file does not match that provided when adding primitive." << std::endl;
    }

    float Ap = context_test.getPrimitivePointer(UUIDp)->getArea();

    if( fabsf(Ap-0.25f*M_PI*sizep.x*sizep.y)/(0.25f*M_PI*sizep.x*sizep.y)>0.01f ){
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

    if( fabsf(area-sizep.x*sizep.y)>0.001f ){
        error_count ++;
        std::cerr << "failed: Patch masked with (u,v) coordinates did not return correct area." << std::endl;
    }

    uv0 = make_vec2( 0, 0 );
    uv1 = make_vec2( 1, 1 );
    uv2 = make_vec2( 0, 1 );

    uint UUIDp3 = context_test.addTriangle( make_vec3(2,3,4), make_vec3(2,3+sizep.y,4), make_vec3(2+sizep.x,3+sizep.y,4), texture, uv0, uv1, uv2 );

    area = context_test.getPrimitivePointer(UUIDp3)->getArea();

    if( fabsf(area-0.5f*0.25f*M_PI*sizep.x*sizep.y)>0.01f ){
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

    uint IDtile = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(3, 1), make_int2(3, 3),
                                             make_SphericalCoord(0, 0));

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
        if( fabsf(gdatad.at(i)-gdatad_io.at(i))>1e-3 ){
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
    for( uint i=filename.size()-4; i<filename.size(); i++ ){
        ext.push_back(filename.at(i));
    }
    if( ext!=".png" ){
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
        for( auto & j : transparencydata ){
            for( bool transparency : j ){
                if( transparency ){
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

std::string Texture::getTextureFile() const{
    return filename;
}

helios::int2 Texture::getSize()const {
    return make_int2(int(transparencydata.front().size()),int(transparencydata.size()));
}

bool Texture::hasTransparencyChannel() const{
    return hastransparencychannel;
}

std::vector<std::vector<bool> >* Texture::getTransparencyData(){
    return &transparencydata;
}

float Texture::getSolidFraction() const{
    return solidfraction;
}

void Context::markGeometryClean(){
    isgeometrydirty = false;
}

void Context::markGeometryDirty(){
    isgeometrydirty = true;
}

bool Context::isGeometryDirty() const{
    return isgeometrydirty;
}

uint Primitive::getUUID() const{
    return UUID;
}

PrimitiveType Primitive::getType() const{
    return prim_type;
}

void Primitive::setParentObjectID(uint objID ){
    parent_object_ID = objID;
}

uint Primitive::getParentObjectID() const{
    return parent_object_ID;
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
    vertices.resize(3);
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

void Primitive::setColor( const helios::RGBcolor& newcolor ){

    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::setColor): Cannot set the color of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    color = make_RGBAcolor(newcolor,1);

}

void Primitive::setColor( const helios::RGBAcolor& newcolor ){

    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::setColor): Cannot set the color of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    color = newcolor;

}

Texture* Primitive::getTexture() const{
    return texture;
}

bool Primitive::hasTexture() const{
    if( texture!=nullptr ){
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

std::vector<vec2> Primitive::getTextureUV(){
    return uv;
}

void Primitive::overrideTextureColor(){

    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::overrideTextureColor): Cannot set the texture options of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    texturecoloroverridden = true;
}

void Primitive::useTextureColor(){

    // if( parent_object_ID!=0 ){
    //   std::cout << "WARNING (Primitive::useTextureColor): Cannot set the texture options of individual primitives within a compound object. Use the setter function for objects." << std::endl;
    //   return;
    // }

    texturecoloroverridden = false;
}

bool Primitive::isTextureColorOverridden() const{
    return texturecoloroverridden;
}

void Primitive::scale( const vec3& S ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Primitive::scale): Cannot scale individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float T[16];
    makeScaleMatrix(S,T);
    matmult(T,transform,transform);
}

void Primitive::translate( const helios::vec3& shift ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Primitive::translate): Cannot translate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float T[16];
    makeTranslationMatrix(shift,T);
    matmult(T,transform,transform);
}

void Patch::rotate( float rot, const char* axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Patch::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

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
        throw( std::runtime_error( "ERROR (Patch::rotate): Rotation axis should be one of x, y, or z." ) );
    }

}

void Patch::rotate( float rot, const helios::vec3& axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Patch::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float R[16];
    makeRotationMatrix(rot,axis,R);
    matmult(R,transform,transform);
}

void Triangle::rotate( float rot, const char* axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Triangle::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

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
        throw( std::runtime_error( "ERROR (Triangle::rotate): Rotation axis should be one of x, y, or z." ) );
    }

}

void Triangle::rotate( float rot, const helios::vec3& axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Triangle::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float R[16];
    makeRotationMatrix(rot,axis,R);
    matmult(R,transform,transform);
}

void Voxel::rotate( float rot, const char* axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Voxel::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float Rz[16];
    makeRotationMatrix(rot,"z",Rz);
    matmult(Rz,transform,transform);

}

void Voxel::rotate( float rot, const helios::vec3& axis ){
    std::cout << "WARNING (Voxel::rotate) - Voxels can only be rotated about the z-axis. Ignoring this call to rotate()." << std::endl;
}

void Triangle::makeTransformationMatrix( const helios::vec3& vert0, const helios::vec3& vert1, const helios::vec3& vert2 ){

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
    /* [0,0] */ V2[0] =  vert0.x;
    /* [0,1] */ V2[1] =  vert1.x;
    /* [0,2] */ V2[2] =  vert2.x;
    /* [0,3] */ V2[3] =  1.f;
    /* [1,0] */ V2[4] =  vert0.y;
    /* [1,1] */ V2[5] =  vert1.y;
    /* [1,2] */ V2[6] =  vert2.y;
    /* [1,3] */ V2[7] =  1.f;
    /* [2,0] */ V2[8] =  vert0.z;
    /* [2,1] */ V2[9] =  vert1.z;
    /* [2,2] */ V2[10] = vert2.z;
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
                transform[ j+i*4 ] += V2[k+i*4]*float(invV1[j+k*4]);
            }
        }
    }

}

float Triangle::getSolidFraction() const{
    return solid_fraction;
}

void Primitive::setPrimitiveData( const char* label, const int& data ){
    std::vector<int> vec{data};
    primitive_data_int[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_INT;
}

void Primitive::setPrimitiveData( const char* label, const uint& data ){
    std::vector<uint> vec{data};
    primitive_data_uint[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_UINT;
}

void Primitive::setPrimitiveData( const char* label, const float& data ){
    std::vector<float> vec{data};
    primitive_data_float[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_FLOAT;
}

void Primitive::setPrimitiveData( const char* label, const double& data ){
    std::vector<double> vec{data};
    primitive_data_double[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_DOUBLE;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec2& data ){
    std::vector<vec2> vec{data};
    primitive_data_vec2[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_VEC2;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec3& data ){
    std::vector<vec3> vec{data};
    primitive_data_vec3[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_VEC3;
}

void Primitive::setPrimitiveData( const char* label, const helios::vec4& data ){
    std::vector<vec4> vec{data};
    primitive_data_vec4[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_VEC4;
}

void Primitive::setPrimitiveData( const char* label, const helios::int2& data ){
    std::vector<int2> vec{data};
    primitive_data_int2[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_INT2;
}

void Primitive::setPrimitiveData( const char* label, const helios::int3& data ){
    std::vector<int3> vec{data};
    primitive_data_int3[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_INT3;
}

void Primitive::setPrimitiveData( const char* label, const helios::int4& data ){
    std::vector<int4> vec{data};
    primitive_data_int4[label] = vec;
    primitive_data_types[label] = HELIOS_TYPE_INT4;
}

void Primitive::setPrimitiveData( const char* label, const std::string& data ){
    std::vector<std::string> vec{data};
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

        auto* data_ptr = (float*)data;

        std::vector<float> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_float[label] = vec;

    }else if( type==HELIOS_TYPE_DOUBLE ){

        auto* data_ptr = (double*)data;

        std::vector<double> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_double[label] = vec;

    }else if( type==HELIOS_TYPE_VEC2 ){

        auto* data_ptr = (vec2*)data;

        std::vector<vec2> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_vec2[label] = vec;

    }else if( type==HELIOS_TYPE_VEC3 ){

        auto* data_ptr = (vec3*)data;

        std::vector<vec3> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_vec3[label] = vec;

    }else if( type==HELIOS_TYPE_VEC4 ){

        auto* data_ptr = (vec4*)data;

        std::vector<vec4> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_vec4[label] = vec;

    }else if( type==HELIOS_TYPE_INT2 ){

        auto* data_ptr = (int2*)data;

        std::vector<int2> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_int2[label] = vec;

    }else if( type==HELIOS_TYPE_INT3 ){

        auto* data_ptr = (int3*)data;

        std::vector<int3> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_int3[label] = vec;

    }else if( type==HELIOS_TYPE_INT4 ){

        auto* data_ptr = (int4*)data;

        std::vector<int4> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        primitive_data_int4[label] = vec;

    }else if( type==HELIOS_TYPE_STRING ){

        auto* data_ptr = (std::string*)data;

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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT ){
        std::vector<int> d = primitive_data_int.at(label);
        data = d.at(0);
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT ){
        std::vector<int> d = primitive_data_int.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, uint& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_UINT ){
        std::vector<uint> d = primitive_data_uint.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type uint, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type uint." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<uint>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_UINT ){
        std::vector<uint> d = primitive_data_uint.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type uint, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type uint." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, float& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_FLOAT ){
        std::vector<float> d = primitive_data_float.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type float, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type float." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<float>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_FLOAT ){
        std::vector<float> d = primitive_data_float.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type float, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type float." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, double& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> d = primitive_data_double.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type double, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type double." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<double>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> d = primitive_data_double.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type double, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type double." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, vec2& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> d = primitive_data_vec2.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type vec2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec2." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<vec2>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> d = primitive_data_vec2.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type vec2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec2." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, vec3& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> d = primitive_data_vec3.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type vec3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec3." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<vec3>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> d = primitive_data_vec3.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type vec3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec3." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, vec4& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> d = primitive_data_vec4.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type vec4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec4." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<vec4>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> d = primitive_data_vec4.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type vec4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type vec4." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, int2& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT2 ){
        std::vector<int2> d = primitive_data_int2.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int2." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int2>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT2 ){
        std::vector<int2> d = primitive_data_int2.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int2, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int2." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, int3& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT3 ){
        std::vector<int3> d = primitive_data_int3.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int3." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int3>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT3 ){
        std::vector<int3> d = primitive_data_int3.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int3, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int3." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, int4& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT4 ){
        std::vector<int4> d = primitive_data_int4.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int4." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<int4>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT4 ){
        std::vector<int4> d = primitive_data_int4.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type int4, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type int4." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::string& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_STRING ){
        std::vector<std::string> d = primitive_data_string.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type string, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type string." ) );
    }

}

void Primitive::getPrimitiveData( const char* label, std::vector<std::string>& data ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_STRING ){
        std::vector<std::string> d = primitive_data_string.at(label);
        data = d;
    }else{
        throw( std::runtime_error( "ERROR (getPrimitiveData): Attempted to get data for type string, but data " + std::string(label) + " for primitive " + std::to_string(UUID) + " does not have type string." ) );
    }

}

HeliosDataType Primitive::getPrimitiveDataType( const char* label ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
    }

    return primitive_data_types.at(label);

}

uint Primitive::getPrimitiveDataSize( const char* label ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive" + std::to_string(UUID) ) );
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

void Primitive::clearPrimitiveData( const char* label ){

    if( !doesPrimitiveDataExist( label ) ){
        return;
    }

    HeliosDataType type = primitive_data_types.at(label);

    if( type==HELIOS_TYPE_INT ){
        primitive_data_int.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_UINT ){
        primitive_data_uint.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_FLOAT ){
        primitive_data_float.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_DOUBLE ){
        primitive_data_double.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_VEC2 ){
        primitive_data_vec2.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_VEC3 ){
        primitive_data_vec3.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_VEC4 ){
        primitive_data_vec4.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_INT2 ){
        primitive_data_int2.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_INT3 ){
        primitive_data_int3.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_INT4 ){
        primitive_data_int4.erase(label);
        primitive_data_types.erase(label);
    }else if( type==HELIOS_TYPE_STRING ){
        primitive_data_string.erase(label);
        primitive_data_types.erase(label);
    }else{
        assert(false);
    }

}

bool Primitive::doesPrimitiveDataExist( const char* label ) const{

    if( primitive_data_types.find(label) == primitive_data_types.end() ){
        return false;
    }else{
        return true;
    }

}

std::vector<std::string> Primitive::listPrimitiveData() const{

    std::vector<std::string> labels(primitive_data_types.size());

    size_t i=0;
    for(const auto & primitive_data_type : primitive_data_types){
        labels.at(i) = primitive_data_type.first;
        i++;
    }

    return labels;

}

void Context::setPrimitiveData( const uint& UUID, const char* label, const int& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const uint& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const float& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const double& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::vec2& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::vec3& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::vec4& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::int2& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::int3& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const helios::int4& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, const std::string& data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,data);
}

void Context::setPrimitiveData( const uint& UUID, const char* label, HeliosDataType type, uint size, void* data ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error( "ERROR (setPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    primitives.at(UUID)->setPrimitiveData(label,type,size,data);
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const int& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const uint& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const float& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const double& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec2& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec3& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::vec4& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int2& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int3& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const helios::int4& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<uint>& UUIDs, const char* label, const std::string& data ){
    for( uint UUID : UUIDs){
        setPrimitiveData( UUID, label, data );
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const int& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const uint& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const float& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const double& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec2& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec3& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::vec4& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int2& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int3& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const helios::int4& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<uint> >& UUIDs, const char* label, const std::string& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const int& data ){
    for(const auto & j : UUIDs){
        for( const auto& UUID : j ){
            setPrimitiveData( UUID, label, data );
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const uint& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const float& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const double& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec2& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec3& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::vec4& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int2& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int3& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const helios::int4& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::setPrimitiveData( const std::vector<std::vector<std::vector<uint> > >& UUIDs, const char* label, const std::string& data ){
    for(const auto & j : UUIDs){
        for( const auto& i : j ) {
            for (const auto &UUID: i) {
                setPrimitiveData(UUID, label, data);
            }
        }
    }
}

void Context::getPrimitiveData(uint UUID, const char* label, int& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, uint& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<uint>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, float& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<float>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, double& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<double>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec2& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec2>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec3& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec3>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, vec4& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<vec4>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int2& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int2>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int3& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int3>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, int4& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<int4>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::string& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

void Context::getPrimitiveData( const uint UUID, const char* label, std::vector<std::string>& data ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->getPrimitiveData(label,data);
}

HeliosDataType Context::getPrimitiveDataType( const uint UUID, const char* label )const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    return primitives.at(UUID)->getPrimitiveDataType(label);
}

uint Context::getPrimitiveDataSize( const uint UUID, const char* label )const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    return primitives.at(UUID)->getPrimitiveDataSize(label);
}

bool Context::doesPrimitiveDataExist( const uint UUID, const char* label ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    return primitives.at(UUID)->doesPrimitiveDataExist(label);
}

void Context::clearPrimitiveData( const uint UUID, const char* label ){
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
    }
    primitives.at(UUID)->clearPrimitiveData(label);
}

void Context::clearPrimitiveData( const std::vector<uint>& UUIDs, const char* label ){
    for( unsigned int UUID : UUIDs){
        if( primitives.find(UUID) == primitives.end() ){
            throw( std::runtime_error("ERROR (getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context."));
        }
        primitives.at(UUID)->clearPrimitiveData(label);
    }
}

//-----

void Context::setObjectData( const uint objID, const char* label, const int& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const uint& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const float& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const double& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::vec2& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::vec3& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::vec4& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::int2& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::int3& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const helios::int4& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, const std::string& data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,data);
}

void Context::setObjectData( const uint objID, const char* label, HeliosDataType type, uint size, void* data ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (setObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->setObjectData(label,type,size,data);
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const int& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const uint& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const float& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const double& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec2& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec3& data ){
    for(unsigned int objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::vec4& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int2& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int3& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const helios::int4& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<uint>& objIDs, const char* label, const std::string& data ){
    for( uint objID : objIDs){
        setObjectData( objID, label, data );
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const int& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const uint& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const float& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const double& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec2& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec3& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::vec4& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int2& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int3& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const helios::int4& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<uint> >& objIDs, const char* label, const std::string& data ){
    for( const auto & j : objIDs){
        for( const auto & objID : j ){
            setObjectData( objID, label, data );
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const int& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const uint& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const float& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const double& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec2& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec3& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::vec4& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int2& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int3& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const helios::int4& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::setObjectData( const std::vector<std::vector<std::vector<uint> > >& objIDs, const char* label, const std::string& data ){
    for( const auto & j : objIDs){
        for( const auto & i : j ){
            for( const auto & objID : i ) {
                setObjectData(objID, label, data);
            }
        }
    }
}

void Context::getObjectData( const uint objID, const char* label, int& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, uint& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<uint>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, float& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<float>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, double& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<double>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, vec2& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<vec2>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, vec3& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<vec3>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, vec4& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<vec4>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, int2& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int2>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, int3& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int3>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, int4& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<int4>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::string& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

void Context::getObjectData( const uint objID, const char* label, std::vector<std::string>& data ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->getObjectData(label,data);
}

HeliosDataType Context::getObjectDataType( const uint objID, const char* label )const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectDataType): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    return objects.at(objID)->getObjectDataType(label);
}

uint Context::getObjectDataSize( const uint objID, const char* label )const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectDataSize): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    return objects.at(objID)->getObjectDataSize(label);
}

bool Context::doesObjectDataExist( const uint objID, const char* label ) const{
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (doesObjectDataExist): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    return objects.at(objID)->doesObjectDataExist(label);
}

void Context::clearObjectData( const uint objID, const char* label ){
    if( objects.find(objID) == objects.end() ){
        throw( std::runtime_error("ERROR (clearObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
    }
    objects.at(objID)->clearObjectData(label);
}

void Context::clearObjectData( const std::vector<uint>& objIDs, const char* label ){
    for( uint objID : objIDs ){
        if( objects.find(objID) == objects.end() ){
            throw( std::runtime_error("ERROR (getObjectData): objID of " + std::to_string(objID) + " does not exist in the Context."));
        }
        objects.at(objID)->clearObjectData(label);
    }
}

void Context::setGlobalData( const char* label, const int& data ){
    std::vector<int> vec{data};
    globaldata[label].type = HELIOS_TYPE_INT;
    globaldata[label].size = 1;
    globaldata[label].global_data_int = vec;
}

void Context::setGlobalData( const char* label, const uint& data ){
    std::vector<uint> vec{data};
    globaldata[label].type = HELIOS_TYPE_UINT;
    globaldata[label].size = 1;
    globaldata[label].global_data_uint = vec;
}

void Context::setGlobalData( const char* label, const float& data ){
    std::vector<float> vec{data};
    globaldata[label].type = HELIOS_TYPE_FLOAT;
    globaldata[label].size = 1;
    globaldata[label].global_data_float = vec;
}

void Context::setGlobalData( const char* label, const double& data ){
    std::vector<double> vec{data};
    globaldata[label].type = HELIOS_TYPE_DOUBLE;
    globaldata[label].size = 1;
    globaldata[label].global_data_double = vec;
}

void Context::setGlobalData( const char* label, const helios::vec2& data ){
    std::vector<vec2> vec{data};
    globaldata[label].type = HELIOS_TYPE_VEC2;
    globaldata[label].size = 1;
    globaldata[label].global_data_vec2 = vec;
}

void Context::setGlobalData( const char* label, const helios::vec3& data ){
    std::vector<vec3> vec{data};
    globaldata[label].type = HELIOS_TYPE_VEC3;
    globaldata[label].size = 1;
    globaldata[label].global_data_vec3 = vec;
}

void Context::setGlobalData( const char* label, const helios::vec4& data ){
    std::vector<vec4> vec{data};
    globaldata[label].type = HELIOS_TYPE_VEC4;
    globaldata[label].size = 1;
    globaldata[label].global_data_vec4 = vec;
}

void Context::setGlobalData( const char* label, const helios::int2& data ){
    std::vector<int2> vec{data};
    globaldata[label].type = HELIOS_TYPE_INT2;
    globaldata[label].size = 1;
    globaldata[label].global_data_int2 = vec;
}

void Context::setGlobalData( const char* label, const helios::int3& data ){
    std::vector<int3> vec{data};
    globaldata[label].type = HELIOS_TYPE_INT3;
    globaldata[label].size = 1;
    globaldata[label].global_data_int3 = vec;
}

void Context::setGlobalData( const char* label, const helios::int4& data ){
    std::vector<int4> vec{data};
    globaldata[label].type = HELIOS_TYPE_INT4;
    globaldata[label].size = 1;
    globaldata[label].global_data_int4 = vec;
}

void Context::setGlobalData( const char* label, const std::string& data ){
    std::vector<std::string> vec{data};
    globaldata[label].type = HELIOS_TYPE_STRING;
    globaldata[label].size = 1;
    globaldata[label].global_data_string = vec;
}

void Context::setGlobalData( const char* label, HeliosDataType type, size_t size, void* data ){

    globaldata[label].type = type;
    globaldata[label].size = size;

    if( type==HELIOS_TYPE_INT ){

        auto* data_ptr = (int*)data;

        std::vector<int> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_int = vec;

    }else if( type==HELIOS_TYPE_UINT ){

        auto* data_ptr = (uint*)data;

        std::vector<uint> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_uint = vec;

    }else if( type==HELIOS_TYPE_FLOAT ){

        auto* data_ptr = (float*)data;

        std::vector<float> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_float = vec;

    }else if( type==HELIOS_TYPE_DOUBLE ){

        auto* data_ptr = (double*)data;

        std::vector<double> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_double = vec;

    }else if( type==HELIOS_TYPE_VEC2 ){

        auto* data_ptr = (vec2*)data;

        std::vector<vec2> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_vec2 = vec;

    }else if( type==HELIOS_TYPE_VEC3 ){

        auto* data_ptr = (vec3*)data;

        std::vector<vec3> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_vec3= vec;

    }else if( type==HELIOS_TYPE_VEC4 ){

        auto* data_ptr = (vec4*)data;

        std::vector<vec4> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_vec4 = vec;

    }else if( type==HELIOS_TYPE_INT2 ){

        auto* data_ptr = (int2*)data;

        std::vector<int2> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_int2 = vec;

    }else if( type==HELIOS_TYPE_INT3 ){

        auto* data_ptr = (int3*)data;

        std::vector<int3> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_int3 = vec;

    }else if( type==HELIOS_TYPE_INT4 ){

        auto* data_ptr = (int4*)data;

        std::vector<int4> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        globaldata[label].global_data_int4 = vec;

    }else if( type==HELIOS_TYPE_STRING ){

        auto* data_ptr = (std::string*)data;

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
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT ){
        std::vector<int> d = gdata.global_data_int;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int, but data '" + std::string(label) + "' does not have type int."));
    }

}

void Context::getGlobalData( const char* label, std::vector<int>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT ){
        std::vector<int> d = gdata.global_data_int;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int, but data '" + std::string(label) + "' does not have type int."));
    }

}

void Context::getGlobalData( const char* label, uint& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_UINT ){
        std::vector<uint> d = gdata.global_data_uint;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type uint, but data '" + std::string(label) + "' does not have type uint."));
    }

}

void Context::getGlobalData( const char* label, std::vector<uint>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_UINT ){
        std::vector<uint> d = gdata.global_data_uint;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type uint, but data '" + std::string(label) + "' does not have type uint."));
    }

}

void Context::getGlobalData( const char* label, float& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_FLOAT ){
        std::vector<float> d = gdata.global_data_float;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type float, but data '" + std::string(label) + "' does not have type float."));
    }

}

void Context::getGlobalData( const char* label, std::vector<float>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_FLOAT ){
        std::vector<float> d = gdata.global_data_float;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type float, but data '" + std::string(label) + "' does not have type float."));
    }

}

void Context::getGlobalData( const char* label, double& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> d = gdata.global_data_double;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type double, but data '" + std::string(label) + "' does not have type double."));
    }

}

void Context::getGlobalData( const char* label, std::vector<double>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_DOUBLE ){
        std::vector<double> d = gdata.global_data_double;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type double, but data '" + std::string(label) + "' does not have type double."));
    }

}

void Context::getGlobalData( const char* label, helios::vec2& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> d = gdata.global_data_vec2;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type vec2, but data '" + std::string(label) + "' does not have type vec2."));
    }

}

void Context::getGlobalData( const char* label, std::vector<helios::vec2>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> d = gdata.global_data_vec2;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type vec2, but data '" + std::string(label) + "' does not have type vec2."));
    }

}

void Context::getGlobalData( const char* label, helios::vec3& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> d = gdata.global_data_vec3;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type vec3, but data '" + std::string(label) + "' does not have type vec3."));
    }

}

void Context::getGlobalData( const char* label, std::vector<helios::vec3>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> d = gdata.global_data_vec3;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type vec3, but data '" + std::string(label) + "' does not have type vec3."));
    }

}

void Context::getGlobalData( const char* label, helios::vec4& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> d = gdata.global_data_vec4;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type vec4, but data '" + std::string(label) + "' does not have type vec4."));
    }

}

void Context::getGlobalData( const char* label, std::vector<helios::vec4>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> d = gdata.global_data_vec4;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type vec4, but data '" + std::string(label) + "' does not have type vec4."));
    }

}

void Context::getGlobalData( const char* label, helios::int2& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT2 ){
        std::vector<int2> d = gdata.global_data_int2;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int2, but data '" + std::string(label) + "' does not have type int2."));
    }

}

void Context::getGlobalData( const char* label, std::vector<helios::int2>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT2 ){
        std::vector<int2> d = gdata.global_data_int2;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int2, but data '" + std::string(label) + "' does not have type int2."));
    }

}

void Context::getGlobalData( const char* label, helios::int3& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT3 ){
        std::vector<int3> d = gdata.global_data_int3;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int3, but data '" + std::string(label) + "' does not have type int3."));
    }

}

void Context::getGlobalData( const char* label, std::vector<helios::int3>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT3 ){
        std::vector<int3> d = gdata.global_data_int3;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int3, but data '" + std::string(label) + "' does not have type int3."));
    }

}

void Context::getGlobalData( const char* label, helios::int4& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT4 ){
        std::vector<int4> d = gdata.global_data_int4;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int4, but data '" + std::string(label) + "' does not have type int4."));
    }

}

void Context::getGlobalData( const char* label, std::vector<helios::int4>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_INT4 ){
        std::vector<int4> d = gdata.global_data_int4;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type int4, but data '" + std::string(label) + "' does not have type int4."));
    }

}

void Context::getGlobalData( const char* label, std::string& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_STRING ){
        std::vector<std::string> d = gdata.global_data_string;
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type string, but data '" + std::string(label) + "' does not have type string."));
    }

}

void Context::getGlobalData( const char* label, std::vector<std::string>& data ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalData): Global data " + std::string(label) + " does not exist in the Context."));
    }

    GlobalData gdata = globaldata.at(label);

    if( gdata.type==HELIOS_TYPE_STRING ){
        std::vector<std::string> d = gdata.global_data_string;
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getGlobalData): Attempted to get global data for type string, but data '" + std::string(label) + "' does not have type string."));
    }

}

HeliosDataType Context::getGlobalDataType( const char* label ) const{

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalDataType): Global data " + std::string(label) + " does not exist in the Context."));
    }

    return globaldata.at(label).type;

}

size_t Context::getGlobalDataSize(const char *label) const {

    if( !doesGlobalDataExist( label ) ){
        throw( std::runtime_error("ERROR (getGlobalDataSize): Global data " + std::string(label) + " does not exist in the Context."));
    }

    return globaldata.at(label).size;

}

bool Context::doesGlobalDataExist( const char* label ) const{

    if( globaldata.find(label) == globaldata.end() ){
        return false;
    }else{
        return true;
    }

}


Patch::Patch( const RGBAcolor& newcolor, uint newUUID ){

    makeIdentityMatrix( transform );

    color = newcolor;
    assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
    UUID = newUUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    solid_fraction = 1.f;
    texture = nullptr;
    texturecoloroverridden = false;

}

Patch::Patch( Texture* newtexture, uint newUUID ){

    makeIdentityMatrix( transform );

    UUID = newUUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    texture = newtexture;
    solid_fraction = texture->getSolidFraction();
    texturecoloroverridden = false;

}

Patch::Patch( Texture* _texture_, const std::vector<vec2>& _uv_, float _solid_fraction_, uint newUUID ){

    makeIdentityMatrix( transform );

    UUID = newUUID;
    prim_type = PRIMITIVE_TYPE_PATCH;

    texture = _texture_;
    uv = _uv_;
    solid_fraction = _solid_fraction_;
    texturecoloroverridden = false;

}

helios::vec2 Patch::getSize() const{
    std::vector<vec3> vertices = getVertices();
    float l = (vertices.at(1)-vertices.at(0)).magnitude();
    float w = (vertices.at(3)-vertices.at(0)).magnitude();
    return make_vec2(l,w);
}

helios::vec3 Patch::getCenter() const{
    return make_vec3(transform[3],transform[7],transform[11]);
}

Triangle::Triangle(  const vec3& _vertex0_, const vec3& _vertex1_, const vec3& _vertex2_, const RGBAcolor& _color_, uint newUUID ){

    makeTransformationMatrix(_vertex0_,_vertex1_,_vertex2_);
    color = _color_;
    UUID = newUUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;
    texture = nullptr;
    solid_fraction = 1.f;
    texturecoloroverridden = false;

}

Triangle::Triangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2, Texture* _texture_, const std::vector<vec2>& _uv_, float _solid_fraction_, uint newUUID ){

    makeTransformationMatrix(vertex0,vertex1,vertex2);
    color = make_RGBAcolor(RGB::red,1);
    UUID = newUUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;

    texture = _texture_;
    uv = _uv_;
    solid_fraction = _solid_fraction_;
    texturecoloroverridden = false;

}

vec3 Triangle::getVertex( int number ){

    if( number<0 || number>2 ){
        throw( std::runtime_error("ERROR (getVertex): vertex index must be 1, 2, or 3."));
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

Voxel::Voxel( const RGBAcolor& _color_, uint newUUID ){

    makeIdentityMatrix(transform);

    color = _color_;
    assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
    UUID = newUUID;
    prim_type = PRIMITIVE_TYPE_VOXEL;
    texture = nullptr;
    texturecoloroverridden = false;

}

float Voxel::getVolume(){

    vec3 size = getSize();

    return size.x*size.y*size.z;
}

vec3 Voxel::getCenter(){

    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;

}

vec3 Voxel::getSize(){

    vec3 n0(0,0,0), nx(1,0,0), ny(0,1,0), nz(0,0,1);
    vec3 n0_T, nx_T, ny_T, nz_T;

    vecmult(transform,n0,n0_T);
    vecmult(transform,nx,nx_T);
    vecmult(transform,ny,ny_T);
    vecmult(transform,nz,nz_T);

    float x = (nx_T-n0_T).magnitude();
    float y = (ny_T-n0_T).magnitude();
    float z = (nz_T-n0_T).magnitude();

    return make_vec3(x,y,z);

}

void Context::setDate( int day, int month, int year ){

    if( day<1 || day>31 ){
        throw( std::runtime_error("ERROR (setDate): Day of month is out of range (day of " + std::to_string(day) + " was given).") );
    }else if( month<1 || month>12){
        throw( std::runtime_error("ERROR (setDate): Month of year is out of range (month of " + std::to_string(month) + " was given).") );
    }else if( year<1000 ){
        throw( std::runtime_error("ERROR (setDate): Year should be specified in YYYY format.") );
    }

    sim_date = make_Date(day,month,year);

}

void Context::setDate( Date date ){

    if( date.day<1 || date.day>31 ){
        throw( std::runtime_error("ERROR (setDate): Day of month is out of range (day of " + std::to_string(date.day) + " was given).") );
    }else if( date.month<1 || date.month>12){
        throw( std::runtime_error("ERROR (setDate): Month of year is out of range (month of " + std::to_string(date.month) + " was given).") );
    }else if( date.year<1000 ){
        throw( std::runtime_error("ERROR (setDate): Year should be specified in YYYY format.") );
    }

    sim_date = date;

}

void Context::setDate( int Julian_day, int year ){

    if( Julian_day<1 || Julian_day>366 ){
        throw( std::runtime_error("ERROR (setDate): Julian day out of range.") );
    }else if( year<1000 ){
        throw( std::runtime_error("ERROR (setDate): Year should be specified in YYYY format.") );
    }

    sim_date = CalendarDay( Julian_day, year );

}

Date Context::getDate() const{
    return sim_date;
}

const char* Context::getMonthString() const{
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

int Context::getJulianDate() const{
    return JulianDay( sim_date.day, sim_date.month, sim_date.year );
}

void Context::setTime( int minute, int hour ){
    setTime(0,minute,hour);
}

void Context::setTime( int second, int minute, int hour ){

    if( second<0 || second>59 ){
        throw( std::runtime_error("ERROR (setTime): Second out of range (0-59).") );
    }else if( minute<0 || minute>59 ){
        throw( std::runtime_error("ERROR (setTime): Minute out of range (0-59).") );
    }else if( hour<0 || hour>23 ){
        throw( std::runtime_error("ERROR (setTime): Hour out of range (0-23).") );
    }

    sim_time = make_Time(hour,minute,second);

}

void Context::setTime( Time time ){

    if( time.minute<0 || time.minute>59 ){
        throw( std::runtime_error("ERROR (setTime): Minute out of range (0-59).") );
    }else if( time.hour<0 || time.hour>23 ){
        throw( std::runtime_error("ERROR (setTime): Hour out of range (0-23).") );
    }

    sim_time = time;

}

Time Context::getTime() const{
    return sim_time;
}

float Context::randu(){
    return unif_distribution(generator);
}

float Context::randu( float minrange, float maxrange ){
    if( maxrange<minrange ){
        throw( std::runtime_error("ERROR (randu): Maximum value of range must be greater than minimum value of range.") );
    }else if( maxrange==minrange ){
        return minrange;
    }else{
        return minrange+unif_distribution(generator)*(maxrange-minrange);
    }
}

int Context::randu( int minrange, int maxrange ){
    if( maxrange<minrange ){
        throw( std::runtime_error("ERROR (randu): Maximum value of range must be greater than minimum value of range.") );
    }else if( maxrange==minrange ){
        return minrange;
    }else{
        return minrange+(int)lroundf(unif_distribution(generator)*float(maxrange-minrange));
    }
}

float Context::randn(){
    return norm_distribution(generator);
}

float Context::randn( float mean, float stddev ){
    return mean+norm_distribution(generator)*fabsf(stddev);
}

uint Context::addPatch(){
    return addPatch(make_vec3(0,0,0),make_vec2(1,1),make_SphericalCoord(0,0),make_RGBAcolor(0,0,0,1));
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

    if( size.x==0 || size.y==0 ){
        throw( std::runtime_error("ERROR (addPatch): Size of patch must be greater than 0.") );
    }

    auto* patch_new = (new Patch( color, currentUUID ));

    patch_new->setParentObjectID(0);

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

    auto* patch_new = (new Patch( texture, currentUUID ));

    patch_new->setParentObjectID(0);

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

    if( size.x==0 || size.y==0 ){
        throw( std::runtime_error("ERROR (addPatch): Size of patch must be greater than 0.") );
    }

    if( uv_center.x-0.5*uv_size.x<-1e-3 || uv_center.y-0.5*uv_size.y<-1e-3 || uv_center.x+0.5*uv_size.x-1.f>1e-3 || uv_center.y+0.5*uv_size.y-1.f>1e-3 ){
        throw( std::runtime_error("ERROR (addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1.") );
    }

    Texture* texture = addTexture( texture_file );

    std::vector<helios::vec2> uv;
    uv.resize(4);
    uv.at(0) = uv_center+make_vec2(-0.5f*uv_size.x,-0.5f*uv_size.y);
    uv.at(1) = uv_center+make_vec2(+0.5f*uv_size.x,-0.5f*uv_size.y);
    uv.at(2) =  uv_center+make_vec2(+0.5f*uv_size.x,+0.5f*uv_size.y);
    uv.at(3) =  uv_center+make_vec2(-0.5f*uv_size.x,+0.5f*uv_size.y);

    float solid_fraction;
    if( texture->hasTransparencyChannel() ){
        std::vector<std::vector<bool> >* alpha = texture->getTransparencyData();
        int A = 0;
        int At = 0;
        int2 sz = texture->getSize();
        int2 uv_min( floor(uv.at(0).x*float(sz.x)), floor(uv.at(0).y*float(sz.y)) );
        int2 uv_max( floor(uv.at(2).x*float(sz.x)), floor(uv.at(2).y*float(sz.y)) );
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
    auto* patch_new = (new Patch( texture, uv, solid_fraction, currentUUID ));

    patch_new->setParentObjectID(0);

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

bool edgeFunction(const helios::vec2 &a, const helios::vec2 &b, const helios::vec2 &c){
    return ((c.y - a.y) * (b.x - a.x)-(c.x - a.x) * (b.y - a.y) >= 0);
}

uint Context::addTriangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2, const RGBAcolor& color ){

    auto* tri_new = (new Triangle( vertex0, vertex1, vertex2, color, currentUUID ));

    tri_new->setParentObjectID(0);
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
        int2 uv_min( (int)lroundf(fmin(fmin(uv0.x,uv1.x),uv2.x)*float(sz.x)), (int)lroundf(fmin(fmin(uv0.y,uv1.y),uv2.y)*float(sz.y)) );
        int2 uv_max( (int)lroundf(fmax(fmax(uv0.x,uv1.x),uv2.x)*float(sz.x)), (int)lroundf(fmax(fmax(uv0.y,uv1.y),uv2.y)*float(sz.y)) );
        int A = 0;
        int At = 0;
        vec2 xy;
        for( int j=uv_min.y; j<uv_max.y; j++ ){
            for( int i=uv_min.x; i<uv_max.x; i++ ){
                xy.x = float(i+0.5)/float(sz.x-1);
                xy.y = float(j+0.5)/float(sz.y-1);

                bool test_0 = edgeFunction( uv.at(0), uv.at(1), xy);
                bool test_1 = edgeFunction( uv.at(1), uv.at(2), xy );
                bool test_2 = edgeFunction( uv.at(2), uv.at(0), xy ) ;
                uint test_sum =  test_0 + test_1 + test_2;

                if(test_sum == 0 || test_sum == 3){
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

    auto* tri_new = (new Triangle( vertex0, vertex1, vertex2, texture, uv, solid_fraction, currentUUID ));

    tri_new->setParentObjectID(0);
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

    auto* voxel_new = (new Voxel( color, currentUUID ));

    voxel_new->setParentObjectID(0);

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

void Context::translatePrimitive(uint UUID, const vec3& shift ){
    getPrimitivePointer(UUID)->translate(shift);
}

void Context::translatePrimitive( const std::vector<uint>& UUIDs, const vec3& shift ){
    for( uint UUID : UUIDs){
        getPrimitivePointer(UUID)->translate(shift);
    }
}

void Context::rotatePrimitive(uint UUID, float rot, const char* axis ){
    getPrimitivePointer(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive( const std::vector<uint>& UUIDs, float rot, const char* axis ){
    for( uint UUID : UUIDs){
        getPrimitivePointer(UUID)->rotate(rot,axis);
    }
}

void Context::rotatePrimitive(uint UUID, float rot, const helios::vec3& axis ){
    getPrimitivePointer(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive( const std::vector<uint>& UUIDs, float rot, helios::vec3& axis ){
    for( uint UUID : UUIDs){
        getPrimitivePointer(UUID)->rotate(rot,axis);
    }
}

void Context::scalePrimitive(uint UUID, const helios::vec3& S ){
    getPrimitivePointer(UUID)->scale(S);
}

void Context::scalePrimitive( const std::vector<uint>& UUIDs, const helios::vec3& S ){
    for( uint UUID : UUIDs){
        getPrimitivePointer(UUID)->scale(S);
    }
}

void Context::deletePrimitive( const std::vector<uint>& UUIDs ){
    for( uint UUID : UUIDs){
        deletePrimitive( UUID );
    }
}

void Context::deletePrimitive(uint UUID ){

    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (deletePrimitive): UUID of " + std::to_string(UUID) + " not found in the context.") );
    }

    Primitive* prim = primitives.at(UUID);

    std::vector<std::string> pdata = prim->listPrimitiveData();

    delete prim;
    primitives.erase(UUID);

    markGeometryDirty();

}

std::vector<uint> Context::copyPrimitive(const std::vector<uint> &UUIDs ){

    std::vector<uint> UUIDs_copy(UUIDs.size());
    size_t i=0;
    for( uint UUID : UUIDs){
        UUIDs_copy.at(i) = copyPrimitive( UUID );
        i++;
    }

    return UUIDs_copy;

}

uint Context::copyPrimitive( uint UUID ){

    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (copyPrimitive): UUID of " + std::to_string(UUID) + " not found in the context.") );
    }

    PrimitiveType type = primitives.at(UUID)->getType();
    uint parentID = primitives.at(UUID)->getParentObjectID();
    bool textureoverride = primitives.at(UUID)->isTextureColorOverridden();

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
        patch_new->setParentObjectID(parentID);
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
        tri_new->setParentObjectID(parentID);
        primitives[currentUUID] = tri_new;
    }else if( type==PRIMITIVE_TYPE_VOXEL ){
        Voxel* p = getVoxelPointer(UUID);
        Voxel* voxel_new;
        //if( !p->hasTexture() ){
        voxel_new = (new Voxel( p->getColorRGBA(), currentUUID ));
        //}else{
        //  voxel_new = (new Voxel( p->getColorRGBA(), currentUUID ));
        /* \todo Texture-mapped voxels constructor here */
        //}
        float transform[16];
        p->getTransformationMatrix(transform);
        voxel_new->setTransformationMatrix(transform);
        voxel_new->setParentObjectID(parentID);
        primitives[currentUUID] = voxel_new;
    }

    copyPrimitiveData( UUID, currentUUID );

    if( textureoverride ){
        getPrimitivePointer(currentUUID)->overrideTextureColor();
    }

    markGeometryDirty();
    currentUUID++;
    return currentUUID-1;
}

void Context::copyPrimitiveData( uint UUID, uint oldUUID){
    //copy the primitive data
    std::vector<std::string> plabel = getPrimitivePointer(UUID)->listPrimitiveData();
    for(auto & p : plabel){

        HeliosDataType type = getPrimitiveDataType( UUID, p.c_str() );

        if( type==HELIOS_TYPE_INT ){
            std::vector<int> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_UINT ){
            std::vector<uint> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_UINT, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_FLOAT ){
            std::vector<float> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_FLOAT, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_DOUBLE ){
            std::vector<double> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_DOUBLE, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_VEC2 ){
            std::vector<vec2> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_VEC2, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_VEC3 ){
            std::vector<vec3> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_VEC3, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_VEC4 ){
            std::vector<vec4> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_VEC4, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_INT2 ){
            std::vector<int2> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT2, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_INT3 ){
            std::vector<int3> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT3, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_INT4 ){
            std::vector<int4> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_INT4, pdata.size(), &pdata.at(0) );
        }else if( type==HELIOS_TYPE_STRING ){
            std::vector<std::string> pdata;
            getPrimitiveData( UUID, p.c_str(), pdata );
            setPrimitiveData( oldUUID, p.c_str(), HELIOS_TYPE_STRING, pdata.size(), &pdata.at(0) );
        }else{
            assert(false);
        }

    }
}

Primitive* Context::getPrimitivePointer( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPrimitivePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }
    return primitives.at(UUID);
}

bool Context::doesPrimitiveExist(uint UUID ) const{
    return primitives.find(UUID) != primitives.end();
}

Patch* Context::getPatchPointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        throw( std::runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " is not a patch.") );
    }
    return dynamic_cast<Patch*>(primitives.at(UUID));
}
Triangle* Context::getTrianglePointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        throw( std::runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " is not a triangle.") );
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID));
}

Voxel* Context::getVoxelPointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        throw( std::runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " is not a voxel.") );
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID));
}

uint Context::getPrimitiveCount() const{
    return primitives.size();
}

std::vector<uint> Context::getAllUUIDs() const{
    std::vector<uint> UUIDs;
    UUIDs.resize(primitives.size());
    size_t i=0;
    for( auto primitive : primitives){
        UUIDs.at(i) = primitive.first;
        i++;
    }
    return UUIDs;
}

void Context::addTimeseriesData(const char* label, float value, const Date &date, const Time &time ){

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

        auto it_data = timeseries_data[label].begin();
        auto it_datevalue = timeseries_datevalue[label].begin();

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

    throw( std::runtime_error("ERROR (addTimeseriesData): Failed to insert timeseries data for unknown reason.") );

}

void Context::setCurrentTimeseriesPoint(const char* label, uint index ){
    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        throw( std::runtime_error("ERROR (setCurrentTimeseriesPoint): Timeseries variable `" + std::string(label) + "' does not exist."));
    }
    setDate( queryTimeseriesDate( label, index ) );
    setTime( queryTimeseriesTime( label, index ) );
}

float Context::queryTimeseriesData(const char* label, const Date &date, const Time &time ) const{
    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        throw( std::runtime_error("ERROR (setCurrentTimeseriesData): Timeseries variable `" + std::string(label) + "' does not exist."));
    }

    double date_value = floor(date.year*366.25) + date.JulianDay();
    date_value += double(time.hour)/24. + double(time.minute)/1440. + double(time.second)/86400.;

    double tmin = timeseries_datevalue.at(label).front();
    double tmax = timeseries_datevalue.at(label).back();

    if( date_value<tmin ){
        std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the earliest data point in the timeseries." << std::endl;
        return timeseries_data.at(label).front();
    }else if( date_value>tmax ){
        std::cerr << "WARNING (queryTimeseriesData): Timeseries date and time is outside of the range of the data. Using the latest data point in the timeseries." << std::endl;
        return timeseries_data.at(label).back();
    }

    if( timeseries_datevalue.at(label).empty() ){
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
            throw( std::runtime_error("ERROR (queryTimeseriesData): Failed to query timeseries data for unknown reason.") );
        }

        double xminus = timeseries_data.at(label).at(i);
        double xplus = timeseries_data.at(label).at(i+1);

        double tminus = timeseries_datevalue.at(label).at(i);
        double tplus = timeseries_datevalue.at(label).at(i+1);

        return float(xminus + (xplus-xminus)*(date_value-tminus)/(tplus-tminus));

    }

}

float Context::queryTimeseriesData( const char* label, const uint index ) const{

    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        std::cout << "WARNING (getTimeseriesData): Timeseries variable " << label << " does not exist." << std::endl;
    }

    return timeseries_data.at(label).at(index);

}

Time Context::queryTimeseriesTime( const char* label, const uint index ) const{

    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        std::cout << "WARNING (getTimeseriesData): Timeseries variable " << label << " does not exist." << std::endl;
    }

    double dateval = timeseries_datevalue.at(label).at(index);

    int year = floor(floor(dateval)/366.25);
    assert( year>1000 && year<10000 );

    int JD = floor(dateval-floor(double(year)*366.25));
    assert( JD>0 && JD<367 );

    int hour = floor((dateval-floor(dateval))*24.);
    int minute = floor( ((dateval-floor(dateval))*24.-double(hour))*60. );
    int second = (int)lround( ( ( ( dateval - floor(dateval) )*24. - double(hour))*60. - double(minute) )*60.);

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
        throw(std::runtime_error("ERROR (queryTimeseriesDate): Timeseries variable `" + std::string(label) + "' does not exist.") );
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
        throw(std::runtime_error("ERROR (getTimeseriesDate): Timeseries variable `" + std::string(label) + "' does not exist.") );
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

    for( auto primitive : primitives){

        std::vector<vec3> verts = getPrimitivePointer(primitive.first)->getVertices();

        for( auto & vert : verts){
            if( vert.x<xbounds.x ){
                xbounds.x = vert.x;
            }else if( vert.x>xbounds.y ){
                xbounds.y = vert.x;
            }
            if( vert.y<ybounds.x ){
                ybounds.x = vert.y;
            }else if( vert.y>ybounds.y ){
                ybounds.y = vert.y;
            }
            if( vert.z<zbounds.x ){
                zbounds.x = vert.z;
            }else if( vert.z>zbounds.y ){
                zbounds.y = vert.z;
            }
        }

    }

}

void Context::getDomainBoundingBox( const std::vector<uint>& UUIDs, vec2& xbounds, vec2& ybounds, vec2& zbounds ) const{

    xbounds.x = 1e8;
    xbounds.y = -1e8;
    ybounds.x = 1e8;
    ybounds.y = -1e8;
    zbounds.x = 1e8;
    zbounds.y = -1e8;

    for( uint UUID : UUIDs){

        std::vector<vec3> verts = getPrimitivePointer( UUID )->getVertices();

        for( auto & vert : verts){
            if( vert.x<xbounds.x ){
                xbounds.x = vert.x;
            }else if( vert.x>xbounds.y ){
                xbounds.y = vert.x;
            }
            if( vert.y<ybounds.x ){
                ybounds.x = vert.y;
            }else if( vert.y>ybounds.y ){
                ybounds.y = vert.y;
            }
            if( vert.z<zbounds.x ){
                zbounds.x = vert.z;
            }else if( vert.z>zbounds.y ){
                zbounds.y = vert.z;
            }
        }

    }

}

void Context::getDomainBoundingSphere( vec3& center, float& radius ) const{

    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( xbounds, ybounds, zbounds );

    center.x = xbounds.x+0.5f*(xbounds.y - xbounds.x);
    center.y = ybounds.x+0.5f*(ybounds.y - ybounds.x);
    center.z = zbounds.x+0.5f*(zbounds.y - zbounds.x);

    radius = 0.5f*sqrtf( powf(xbounds.y-xbounds.x,2) + powf(ybounds.y-ybounds.x,2) + powf((zbounds.y-zbounds.x),2) );


}

void Context::getDomainBoundingSphere( const std::vector<uint>& UUIDs, vec3& center, float& radius ) const{

    vec2 xbounds, ybounds, zbounds;
    getDomainBoundingBox( UUIDs, xbounds, ybounds, zbounds );

    center.x = xbounds.x+0.5f*(xbounds.y - xbounds.x);
    center.y = ybounds.x+0.5f*(ybounds.y - ybounds.x);
    center.z = zbounds.x+0.5f*(zbounds.y - zbounds.x);

    radius = 0.5f*sqrtf( powf(xbounds.y-xbounds.x,2) + powf(ybounds.y-ybounds.x,2) + powf((zbounds.y-zbounds.x),2) );


}

void Context::cropDomainX(const vec2 &xbounds ){

    std::vector<vec3> vertices;

    std::vector<uint> UUIDs_all = getAllUUIDs();

    for( uint p : UUIDs_all){

        vertices = getPrimitivePointer(p)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.x<xbounds.x || vertex.x>xbounds.y ){
                deletePrimitive( p );
                break;
            }
        }

    }

    if(getPrimitiveCount() == 0 ){
        std::cout << "WARNING (cropDomainX): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }

}

void Context::cropDomainY(const vec2 &ybounds ){

    std::vector<vec3> vertices;

    std::vector<uint> UUIDs_all = getAllUUIDs();

    for( uint p : UUIDs_all){

        vertices = getPrimitivePointer(p)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.y<ybounds.x || vertex.y>ybounds.y ){
                deletePrimitive( p );
                break;
            }
        }

    }

    if(getPrimitiveCount() == 0 ){
        std::cout << "WARNING (cropDomainY): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }

}

void Context::cropDomainZ(const vec2 &zbounds ){

    std::vector<vec3> vertices;

    std::vector<uint> UUIDs_all = getAllUUIDs();

    for( uint p : UUIDs_all){

        vertices = getPrimitivePointer(p)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.z<zbounds.x || vertex.z>zbounds.y ){
                deletePrimitive( p );
                break;
            }
        }

    }

    if(getPrimitiveCount() == 0 ){
        std::cout << "WARNING (cropDomainZ): No primitives were inside cropped area, and thus all primitives were deleted." << std::endl;
    }

}

void Context::cropDomain(const std::vector<uint> &UUIDs, const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds ){

    std::vector<vec3> vertices;

    size_t delete_count = 0;
    for( uint UUID : UUIDs){

        vertices = getPrimitivePointer(UUID)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.x<xbounds.x || vertex.x>xbounds.y ){
                deletePrimitive( UUID );
                delete_count ++;
                break;
            }
            if( vertex.y<ybounds.x || vertex.y>ybounds.y ){
                deletePrimitive( UUID );
                delete_count ++;
                break;
            }
            if( vertex.z<zbounds.x || vertex.z>zbounds.y ){
                deletePrimitive( UUID );
                delete_count ++;
                break;
            }
        }

    }

    if( delete_count==UUIDs.size() ){
        std::cout << "WARNING (cropDomain): No specified primitives were entirely inside cropped area, and thus all specified primitives were deleted." << std::endl;
    }

}

void Context::cropDomain(const vec2 &xbounds, const vec2 &ybounds, const vec2 &zbounds ){
    cropDomain( getAllUUIDs(), xbounds, ybounds, zbounds );
}

uint CompoundObject::getObjectID() const{
    return OID;
}

helios::ObjectType CompoundObject::getObjectType() const{
    return type;
}

uint CompoundObject::getPrimitiveCount() const{
    return UUIDs.size();
}


std::vector<uint> CompoundObject::getPrimitiveUUIDs() const{

    std::vector<uint> U = UUIDs;

    for( uint UUID : U){

        /* \todo More efficient implementation possible using swap-and-pop */
        if( !context->doesPrimitiveExist( UUID ) ){
            U.erase( U.begin()+UUID);
        }
    }

    return U;

}

bool CompoundObject::doesObjectContainPrimitive(uint UUID ){

    return find(UUIDs.begin(),UUIDs.end(),UUID)!=UUIDs.end();

}

helios::vec3 CompoundObject::getObjectCenter() const{

    vec2 xbounds, ybounds, zbounds;

    std::vector<uint> U = getPrimitiveUUIDs();

    context->getDomainBoundingBox( U, xbounds, ybounds, zbounds );

    vec3 origin;

    origin.x = 0.5f*(xbounds.x+xbounds.y);
    origin.y = 0.5f*(ybounds.x+ybounds.y);
    origin.z = 0.5f*(zbounds.x+zbounds.y);

    return origin;
}

float CompoundObject::getArea() const{

    float area = 0.f;

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            area += context->getPrimitivePointer( UUID )->getArea();
        }

    }

    return area;

}

void CompoundObject::setColor( const helios::RGBcolor& a_color ){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->getPrimitivePointer( UUID )->setColor( a_color );
        }

    }

    color = make_RGBAcolor(a_color, 1.f);

}

void CompoundObject::setColor( const helios::RGBAcolor& a_color ){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->getPrimitivePointer( UUID )->setColor( a_color );
        }

    }

    color = a_color;

}

RGBcolor CompoundObject::getColor()const {
    return make_RGBcolor( color.r, color.g, color.b );
}

RGBcolor CompoundObject::getRGBColor()const {
    return make_RGBcolor( color.r, color.g, color.b );
}

RGBAcolor CompoundObject::getRGBAColor()const {
    return color;
}

void CompoundObject::overrideTextureColor(){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->getPrimitivePointer( UUID )->overrideTextureColor();
        }

    }
}

void CompoundObject::useTextureColor(){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->getPrimitivePointer( UUID )->useTextureColor();
        }

    }
}

void CompoundObject::translate( const helios::vec3& shift ){

    float T[16], T_prim[16];
    makeTranslationMatrix(shift,T);

    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);
        }

    }

}

void CompoundObject::rotate( float rot, const char* axis ){

    if( strcmp(axis,"z")==0 ){
        float Rz[16], Rz_prim[16];
        makeRotationMatrix(-rot,"z",Rz);
        matmult(Rz,transform,transform);

        for( uint UUID : UUIDs){
            if( context->doesPrimitiveExist( UUID ) ){
                context->getPrimitivePointer( UUID )->getTransformationMatrix(Rz_prim);
                matmult(Rz,Rz_prim,Rz_prim);
                context->getPrimitivePointer( UUID )->setTransformationMatrix(Rz_prim);
            }
        }
    }else if( strcmp(axis,"y")==0 ){
        float Ry[16], Ry_prim[16];
        makeRotationMatrix(rot,"y",Ry);
        matmult(Ry,transform,transform);
        for( uint UUID : UUIDs){
            if( context->doesPrimitiveExist( UUID ) ){
                context->getPrimitivePointer( UUID )->getTransformationMatrix(Ry_prim);
                matmult(Ry,Ry_prim,Ry_prim);
                context->getPrimitivePointer( UUID )->setTransformationMatrix(Ry_prim);
            }
        }
    }else if( strcmp(axis,"x")==0 ){
        float Rx[16], Rx_prim[16];
        makeRotationMatrix(rot,"x",Rx);
        matmult(Rx,transform,transform);
        for( uint UUID : UUIDs){
            if( context->doesPrimitiveExist( UUID ) ){
                context->getPrimitivePointer( UUID )->getTransformationMatrix(Rx_prim);
                matmult(Rx,Rx_prim,Rx_prim);
                context->getPrimitivePointer( UUID )->setTransformationMatrix(Rx_prim);
            }
        }
    }else{
        throw( std::runtime_error("ERROR (CompoundObject::rotate): Rotation axis should be one of x, y, or z.") );
    }

}

void CompoundObject::rotate( float rot, const helios::vec3& axis ){

    float R[16], R_prim[16];
    makeRotationMatrix(rot,axis,R);
    matmult(R,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(R_prim);
            matmult(R,R_prim,R_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(R_prim);

        }

    }

}

void CompoundObject::getTransformationMatrix( float (&T)[16] ) const{
    for( int i=0; i<16; i++ ){
        T[i]=transform[i];
    }
}

void CompoundObject::setTransformationMatrix( float (&T)[16] ){

    for( int i=0; i<16; i++ ){
        transform[i] = T[i];
    }

}

void CompoundObject::setObjectData( const char* label, const int& data ){
    std::vector<int> vec{data};
    object_data_int[label] = vec;
    object_data_types[label] = HELIOS_TYPE_INT;
}

void CompoundObject::setObjectData( const char* label, const uint& data ){
    std::vector<uint> vec{data};
    object_data_uint[label] = vec;
    object_data_types[label] = HELIOS_TYPE_UINT;
}

void CompoundObject::setObjectData( const char* label, const float& data ){
    std::vector<float> vec{data};
    object_data_float[label] = vec;
    object_data_types[label] = HELIOS_TYPE_FLOAT;
}

void CompoundObject::setObjectData( const char* label, const double& data ){
    std::vector<double> vec{data};
    object_data_double[label] = vec;
    object_data_types[label] = HELIOS_TYPE_DOUBLE;
}

void CompoundObject::setObjectData( const char* label, const helios::vec2& data ){
    std::vector<vec2> vec{data};
    object_data_vec2[label] = vec;
    object_data_types[label] = HELIOS_TYPE_VEC2;
}

void CompoundObject::setObjectData( const char* label, const helios::vec3& data ){
    std::vector<vec3> vec{data};
    object_data_vec3[label] = vec;
    object_data_types[label] = HELIOS_TYPE_VEC3;
}

void CompoundObject::setObjectData( const char* label, const helios::vec4& data ){
    std::vector<vec4> vec{data};
    object_data_vec4[label] = vec;
    object_data_types[label] = HELIOS_TYPE_VEC4;
}

void CompoundObject::setObjectData( const char* label, const helios::int2& data ){
    std::vector<int2> vec{data};
    object_data_int2[label] = vec;
    object_data_types[label] = HELIOS_TYPE_INT2;
}

void CompoundObject::setObjectData( const char* label, const helios::int3& data ){
    std::vector<int3> vec{data};
    object_data_int3[label] = vec;
    object_data_types[label] = HELIOS_TYPE_INT3;
}

void CompoundObject::setObjectData( const char* label, const helios::int4& data ){
    std::vector<int4> vec{data};
    object_data_int4[label] = vec;
    object_data_types[label] = HELIOS_TYPE_INT4;
}

void CompoundObject::setObjectData( const char* label, const std::string& data ){
    std::vector<std::string> vec{data};
    object_data_string[label] = vec;
    object_data_types[label] = HELIOS_TYPE_STRING;
}

void CompoundObject::setObjectData( const char* label, HeliosDataType a_type, uint size, void* data ){

    object_data_types[label] = a_type;

    if( a_type==HELIOS_TYPE_INT ){

        int* data_ptr = (int*)data;

        std::vector<int> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_int[label] = vec;

    }else if( a_type==HELIOS_TYPE_UINT ){

        uint* data_ptr = (uint*)data;

        std::vector<uint> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_uint[label] = vec;

    }else if( a_type==HELIOS_TYPE_FLOAT ){

        auto* data_ptr = (float*)data;

        std::vector<float> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_float[label] = vec;

    }else if( a_type==HELIOS_TYPE_DOUBLE ){

        auto* data_ptr = (double*)data;

        std::vector<double> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_double[label] = vec;

    }else if( a_type==HELIOS_TYPE_VEC2 ){

        auto* data_ptr = (vec2*)data;

        std::vector<vec2> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_vec2[label] = vec;

    }else if( a_type==HELIOS_TYPE_VEC3 ){

        auto* data_ptr = (vec3*)data;

        std::vector<vec3> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_vec3[label] = vec;

    }else if( a_type==HELIOS_TYPE_VEC4 ){

        auto* data_ptr = (vec4*)data;

        std::vector<vec4> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_vec4[label] = vec;

    }else if( a_type==HELIOS_TYPE_INT2 ){

        auto* data_ptr = (int2*)data;

        std::vector<int2> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_int2[label] = vec;

    }else if( a_type==HELIOS_TYPE_INT3 ){

        auto* data_ptr = (int3*)data;

        std::vector<int3> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_int3[label] = vec;

    }else if( a_type==HELIOS_TYPE_INT4 ){

        auto* data_ptr = (int4*)data;

        std::vector<int4> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_int4[label] = vec;

    }else if( a_type==HELIOS_TYPE_STRING ){

        auto* data_ptr = (std::string*)data;

        std::vector<std::string> vec;
        vec.resize(size);
        for( size_t i=0; i<size; i++ ){
            vec.at(i) = data_ptr[i];
        }
        object_data_string[label] = vec;

    }

}

void CompoundObject::getObjectData( const char* label, int& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT ){
        std::vector<int> d = object_data_int.at(label);
        data = d.at(0);
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<int>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT ){
        std::vector<int> d = object_data_int.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int."));
    }

}

void CompoundObject::getObjectData( const char* label, uint& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_UINT ){
        std::vector<uint> d = object_data_uint.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type uint, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type uint."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<uint>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_UINT ){
        std::vector<uint> d = object_data_uint.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type uint, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type uint."));
    }

}

void CompoundObject::getObjectData( const char* label, float& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_FLOAT ){
        std::vector<float> d = object_data_float.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type float, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type float."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<float>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_FLOAT ){
        std::vector<float> d = object_data_float.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type float, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type float."));
    }

}

void CompoundObject::getObjectData( const char* label, double& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_DOUBLE ){
        std::vector<double> d = object_data_double.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type double, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type double."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<double>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_DOUBLE ){
        std::vector<double> d = object_data_double.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type double, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type double."));
    }

}

void CompoundObject::getObjectData( const char* label, vec2& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> d = object_data_vec2.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type vec2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec2."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<vec2>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_VEC2 ){
        std::vector<vec2> d = object_data_vec2.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type vec2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec2."));
    }

}

void CompoundObject::getObjectData( const char* label, vec3& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> d = object_data_vec3.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type vec3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec3."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<vec3>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_VEC3 ){
        std::vector<vec3> d = object_data_vec3.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type vec3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec3."));
    }

}

void CompoundObject::getObjectData( const char* label, vec4& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> d = object_data_vec4.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type vec4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec4."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<vec4>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_VEC4 ){
        std::vector<vec4> d = object_data_vec4.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type vec4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type vec4."));
    }

}

void CompoundObject::getObjectData( const char* label, int2& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT2 ){
        std::vector<int2> d = object_data_int2.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int2."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<int2>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT2 ){
        std::vector<int2> d = object_data_int2.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int2, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int2."));
    }

}

void CompoundObject::getObjectData( const char* label, int3& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT3 ){
        std::vector<int3> d = object_data_int3.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int3."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<int3>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT3 ){
        std::vector<int3> d = object_data_int3.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int3, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int3."));
    }

}

void CompoundObject::getObjectData( const char* label, int4& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT4 ){
        std::vector<int4> d = object_data_int4.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int4."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<int4>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_INT4 ){
        std::vector<int4> d = object_data_int4.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type int4, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type int4."));
    }

}

void CompoundObject::getObjectData( const char* label, std::string& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_STRING ){
        std::vector<std::string> d = object_data_string.at(label);
        data = d.front();
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type string, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type string."));
    }

}

void CompoundObject::getObjectData( const char* label, std::vector<std::string>& data ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    if( object_data_types.at(label)==HELIOS_TYPE_STRING ){
        std::vector<std::string> d = object_data_string.at(label);
        data = d;
    }else{
        throw( std::runtime_error("ERROR (getObjectData): Attempted to get data for type string, but data '" + std::string(label) + "' for object " + std::to_string(OID) + " does not have type string."));
    }

}

HeliosDataType CompoundObject::getObjectDataType( const char* label ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    return object_data_types.at(label);

}

uint CompoundObject::getObjectDataSize( const char* label ) const{

    if( !doesObjectDataExist( label ) ){
        throw( std::runtime_error("ERROR (getObjectData): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID)));
    }

    HeliosDataType qtype = object_data_types.at(label);

    if( qtype==HELIOS_TYPE_INT ){
        return object_data_int.at(label).size();
    }else if( qtype==HELIOS_TYPE_UINT ){
        return object_data_uint.at(label).size();
    }else if( qtype==HELIOS_TYPE_FLOAT ){
        return object_data_float.at(label).size();
    }else if( qtype==HELIOS_TYPE_DOUBLE ){
        return object_data_double.at(label).size();
    }else if( qtype==HELIOS_TYPE_VEC2 ){
        return object_data_vec2.at(label).size();
    }else if( qtype==HELIOS_TYPE_VEC3 ){
        return object_data_vec3.at(label).size();
    }else if( qtype==HELIOS_TYPE_VEC4 ){
        return object_data_vec4.at(label).size();
    }else if( qtype==HELIOS_TYPE_INT2 ){
        return object_data_int2.at(label).size();
    }else if( qtype==HELIOS_TYPE_INT3 ){
        return object_data_int3.at(label).size();
    }else if( qtype==HELIOS_TYPE_INT4 ){
        return object_data_int4.at(label).size();
    }else if( qtype==HELIOS_TYPE_STRING ){
        return object_data_string.at(label).size();
    }else{
        assert( false );
    }

    return 0;

}

void CompoundObject::clearObjectData( const char* label ){

    if( !doesObjectDataExist( label ) ){
        return;
    }

    HeliosDataType qtype = object_data_types.at(label);

    if( qtype==HELIOS_TYPE_INT ){
        object_data_int.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_UINT ){
        object_data_uint.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_FLOAT ){
        object_data_float.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_DOUBLE ){
        object_data_double.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_VEC2 ){
        object_data_vec2.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_VEC3 ){
        object_data_vec3.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_VEC4 ){
        object_data_vec4.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_INT2 ){
        object_data_int2.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_INT3 ){
        object_data_int3.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_INT4 ){
        object_data_int4.erase(label);
        object_data_types.erase(label);
    }else if( qtype==HELIOS_TYPE_STRING ){
        object_data_string.erase(label);
        object_data_types.erase(label);
    }else{
        assert(false);
    }

}

bool CompoundObject::doesObjectDataExist( const char* label ) const{

    if( object_data_types.find(label) == object_data_types.end() ){
        return false;
    }else{
        return true;
    }

}

std::vector<std::string> CompoundObject::listObjectData() const{

    std::vector<std::string> labels(object_data_types.size());

    size_t i=0;
    for(const auto & object_data_type : object_data_types){
        labels.at(i) = object_data_type.first;
        i++;
    }

    return labels;

}

CompoundObject* Context::getObjectPointer( uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return objects.at(ObjID);
}

bool Context::doesObjectExist( const uint ObjID ) const{
    return objects.find(ObjID) != objects.end();
}

std::vector<uint> Context::getAllObjectIDs() const{
    std::vector<uint> objIDs;
    objIDs.resize(objects.size());
    size_t i=0;
    for(auto object : objects){
        objIDs.at(i) = object.first;
        i++;
    }
    return objIDs;
}

void Context::deleteObject(const std::vector<uint> &ObjIDs ){
    for( uint ObjID : ObjIDs){
        deleteObject( ObjID );
    }
}

void Context::deleteObject(uint ObjID ){

    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (deleteObject): Object ID of " + std::to_string(ObjID) + " not found in the context."));
    }

    CompoundObject* obj = objects.at(ObjID);

    std::vector<uint> UUIDs = obj->getPrimitiveUUIDs();
    deletePrimitive(UUIDs);

    delete obj;
    objects.erase(ObjID);

    markGeometryDirty();

}

std::vector<uint> Context::copyObject(const std::vector<uint> &ObjIDs ){

    std::vector<uint> ObjIDs_copy(ObjIDs.size());
    size_t i=0;
    for( uint ObjID : ObjIDs){
        ObjIDs_copy.at(i) = copyObject( ObjID );
        i++;
    }

    return ObjIDs_copy;

}

uint Context::copyObject(uint ObjID ){

    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (copyObject): Object ID of " + std::to_string(ObjID) + " not found in the context."));
    }

    ObjectType type = objects.at(ObjID)->getObjectType();

    std::vector<uint> UUIDs = getObjectPointer(ObjID)->getPrimitiveUUIDs();

    std::vector<uint> UUIDs_copy = copyPrimitive( UUIDs );
    for( uint p : UUIDs_copy){
        getPrimitivePointer(p)->setParentObjectID( currentObjectID );
    }

    if( type==OBJECT_TYPE_TILE ){

        Tile* o = getTileObjectPointer( ObjID );

        int2 subdiv = o->getSubdivisionCount();

        auto* tile_new = (new Tile( currentObjectID, UUIDs_copy, subdiv, this ) );

        objects[currentObjectID] = tile_new;

    }else if( type==OBJECT_TYPE_SPHERE ){

        Sphere* o = getSphereObjectPointer( ObjID );

        uint subdiv = o->getSubdivisionCount();

        auto* sphere_new = (new Sphere( currentObjectID, UUIDs_copy, subdiv, this ) );

        objects[currentObjectID] = sphere_new;

    }else if( type==OBJECT_TYPE_TUBE ){

        Tube* o = getTubeObjectPointer( ObjID );

        std::vector<vec3> nodes = o->getNodes();
        std::vector<float> radius = o->getNodeRadii();
        uint subdiv = o->getSubdivisionCount();

        auto* tube_new = (new Tube( currentObjectID, UUIDs_copy, nodes, radius, subdiv, this ) );

        objects[currentObjectID] = tube_new;

    }else if( type==OBJECT_TYPE_BOX ){

        Box* o = getBoxObjectPointer( ObjID );

        vec3 size = o->getSize();
        int3 subdiv = o->getSubdivisionCount();

        auto* box_new = (new Box( currentObjectID, UUIDs_copy, subdiv, this ) );

        objects[currentObjectID] = box_new;

    }else if( type==OBJECT_TYPE_DISK ){

        Disk* o = getDiskObjectPointer( ObjID );

        vec2 size = o->getSize();
        uint subdiv = o->getSubdivisionCount();

        auto* disk_new = (new Disk( currentObjectID, UUIDs_copy, subdiv, this ) );

        objects[currentObjectID] = disk_new;

    }else if( type==OBJECT_TYPE_POLYMESH ){

        Polymesh* o = getPolymeshObjectPointer( ObjID );

        auto* polymesh_new = (new Polymesh( currentObjectID, UUIDs_copy, this ) );

        objects[currentObjectID] = polymesh_new;

    }else if( type==OBJECT_TYPE_CONE ){

        Cone* o = getConeObjectPointer( ObjID );

        std::vector<vec3> nodes = o->getNodes();
        std::vector<float> radius = o->getNodeRadii();
        uint subdiv = o->getSubdivisionCount();

        auto* cone_new = (new Cone( currentObjectID, UUIDs_copy, nodes.at(0), nodes.at(1), radius.at(0), radius.at(1), subdiv, this ) );

        objects[currentObjectID] = cone_new;

    }

    float T[16];
    getObjectPointer( ObjID )->getTransformationMatrix( T );

    getObjectPointer( currentObjectID )->setTransformationMatrix( T );


    markGeometryDirty();
    currentObjectID++;
    return currentObjectID-1;
}

Tile::Tile(uint a_OID, const std::vector<uint> &a_UUIDs, const int2 &a_subdiv, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_TILE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    context = a_context;

}

Tile* Context::getTileObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Tile*>(objects.at(ObjID));
}

helios::vec2 Tile::getSize() const{
    std::vector<vec3> vertices = getVertices();
    float l = (vertices.at(1)-vertices.at(0)).magnitude();
    float w = (vertices.at(3)-vertices.at(0)).magnitude();
    return make_vec2(l,w);
}

vec3 Tile::getCenter() const{

    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;

}


helios::int2 Tile::getSubdivisionCount() const{
    return subdiv;
}

std::vector<helios::vec3> Tile::getVertices() const{

    std::vector<helios::vec3> vertices;
    vertices.resize(4);

    //subcenter = make_vec3(-0.5*size.x+(float(i)+0.5)*subsize.x,-0.5*size.y+(float(j)+0.5)*subsize.y,0);
    //Y[0] = make_vec3( -0.5f, -0.5f, 0.f);
    //Y[1] = make_vec3( 0.5f, -0.5f, 0.f);
    //Y[2] = make_vec3( 0.5f, 0.5f, 0.f);
    //Y[3] = make_vec3( -0.5f, 0.5f, 0.f);

    vertices.at(0) = context->getPrimitivePointer( UUIDs.front() )->getVertices().at(0);

    vertices.at(1) = context->getPrimitivePointer( UUIDs.at( subdiv.x-1 ) )->getVertices().at(1);

    vertices.at(2) = context->getPrimitivePointer( UUIDs.at( subdiv.x*subdiv.y-1 ) )->getVertices().at(2);

    vertices.at(3) = context->getPrimitivePointer( UUIDs.at( subdiv.x*subdiv.y-subdiv.x ) )->getVertices().at(3);

    return vertices;

}

vec3 Tile::getNormal() const{

    return context->getPatchPointer( UUIDs.front() )->getNormal();

}

std::vector<helios::vec2> Tile::getTextureUV() const{

    std::vector<helios::vec2> uv;
    uv.resize(4);

    uv.at(0) = context->getPrimitivePointer( UUIDs.at( subdiv.x-1 ) )->getTextureUV().at(0);

    uv.at(1) = context->getPrimitivePointer( UUIDs.at( 0) )->getTextureUV().at(1);

    uv.at(2) = context->getPrimitivePointer( UUIDs.at( subdiv.x*subdiv.y-subdiv.y ) )->getTextureUV().at(2);

    uv.at(3) = context->getPrimitivePointer( UUIDs.at( subdiv.x*subdiv.y-1 ) )->getTextureUV().at(3);

    return uv;

}

void Tile::scale(const vec3 &S ){

    float T[16], T_prim[16];
    makeScaleMatrix( S, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);

        }

    }

}

Sphere::Sphere(uint a_OID, const std::vector<uint> &a_UUIDs, uint a_subdiv, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_SPHERE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    context = a_context;

}

Sphere* Context::getSphereObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Sphere*>(objects.at(ObjID));
}

float Sphere::getRadius() const{

    vec3 n0(0,0,0), n1(1,0,0);
    vec3 n0_T, n1_T;

    vecmult(transform,n0,n0_T);
    vecmult(transform,n1,n1_T);

    return  (n1_T-n0_T).magnitude();

}

vec3 Sphere::getCenter() const{

    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;

}

uint Sphere::getSubdivisionCount() const{
    return subdiv;
}

void Sphere::scale( float S ){

    float T[16], T_prim[16];
    makeScaleMatrix( make_vec3(S,S,S), T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);

        }

    }

}


Tube::Tube(uint a_OID, const std::vector<uint> &a_UUIDs, const std::vector<vec3> &a_nodes, const std::vector<float> &a_radius, uint a_subdiv, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_TUBE;
    UUIDs = a_UUIDs;
    nodes = a_nodes;
    radius = a_radius;
    subdiv = a_subdiv;
    context = a_context;

}

Tube* Context::getTubeObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Tube*>(objects.at(ObjID));
}

std::vector<helios::vec3> Tube::getNodes() const{

    std::vector<vec3> nodes_T;
    nodes_T.resize( nodes.size() );

    for( uint i=0; i<nodes.size(); i++ ){
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    return nodes_T;

}

std::vector<float> Tube::getNodeRadii() const{
    std::vector<float> radius_T;
    radius_T.resize(radius.size());
    for( int i=0; i<radius.size(); i++ ){

        vec3 n0(0,0,0), nx(radius.at(i),0,0);
        vec3 n0_T, nx_T;

        vecmult(transform,n0,n0_T);
        vecmult(transform,nx,nx_T);

        radius_T.at(i) = (nx_T-n0_T).magnitude();

    }
    return radius_T;
}

uint Tube::getSubdivisionCount() const{
    return subdiv;
}

void Tube::scale( float S ){

    float T[16], T_prim[16];
    makeScaleMatrix( make_vec3(S,S,S), T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);

        }

    }

}

Box::Box(uint a_OID, const std::vector<uint> &a_UUIDs, const int3 &a_subdiv, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_BOX;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    context = a_context;

}

Box* Context::getBoxObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Box*>(objects.at(ObjID));
}

vec3 Box::getSize() const{

    vec3 n0(0,0,0), nx(1,0,0), ny(0,1,0), nz(0,0,1);

    vec3 n0_T, nx_T, ny_T, nz_T;

    vecmult(transform,n0,n0_T);
    vecmult(transform,nx,nx_T);
    vecmult(transform,ny,ny_T);
    vecmult(transform,nz,nz_T);

    float x = (nx_T-n0_T).magnitude();
    float y = (ny_T-n0_T).magnitude();
    float z = (nz_T-n0_T).magnitude();

    return make_vec3( x, y, z );

}

vec3 Box::getCenter() const{

    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;

}

helios::int3 Box::getSubdivisionCount() const{
    return subdiv;
}

void Box::scale(const vec3 &S ){

    float T[16], T_prim[16];
    makeScaleMatrix( S, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);

        }

    }

}

Disk::Disk(uint a_OID, const std::vector<uint> &a_UUIDs, uint a_subdiv, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_DISK;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    context = a_context;

}

Disk* Context::getDiskObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Disk*>(objects.at(ObjID));
}

vec2 Disk::getSize() const{

    vec3 n0(0,0,0), nx(1,0,0), ny(0,1,0);
    vec3 n0_T, nx_T, ny_T;

    vecmult(transform,n0,n0_T);
    vecmult(transform,nx,nx_T);
    vecmult(transform,ny,ny_T);

    float x = (nx_T-n0_T).magnitude();
    float y = (ny_T-n0_T).magnitude();

    return make_vec2(x,y);

}

vec3 Disk::getCenter() const{

    vec3 center;
    vec3 Y;
    Y.x = 0.f;
    Y.y = 0.f;
    Y.z = 0.f;

    center.x = transform[0] * Y.x + transform[1] * Y.y + transform[2] * Y.z + transform[3];
    center.y = transform[4] * Y.x + transform[5] * Y.y + transform[6] * Y.z + transform[7];
    center.z = transform[8] * Y.x + transform[9] * Y.y + transform[10] * Y.z + transform[11];

    return center;

}

uint Disk::getSubdivisionCount() const{
    return subdiv;
}

void Disk::scale(const vec3 &S ){

    float T[16], T_prim[16];
    makeScaleMatrix( S, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);

        }

    }

}

Polymesh::Polymesh(uint a_OID, const std::vector<uint> &a_UUIDs, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_POLYMESH;
    UUIDs = a_UUIDs;
    context = a_context;

}

Polymesh* Context::getPolymeshObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Polymesh*>(objects.at(ObjID));
}

Cone::Cone(uint a_OID, const std::vector<uint> &a_UUIDs, const vec3 &a_node0, const vec3 &a_node1, float a_radius0, float a_radius1, uint a_subdiv, helios::Context* a_context ){

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_CONE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    context = a_context;
    nodes = {a_node0, a_node1};
    radii = {a_radius0, a_radius1};

}

Cone* Context::getConeObjectPointer( const uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Cone*>(objects.at(ObjID));
}

std::vector<helios::vec3> Cone::getNodes() const{

    std::vector<vec3> nodes_T;

    nodes_T.resize( 2 );

    for( int i=0; i<2; i++ ){
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    return nodes_T;
}

helios::vec3 Cone::getNode( int number ) const{

    if( number<0 || number>1 ){
        throw( std::runtime_error("ERROR (Cone::getNode): node number must be 0 or 1."));
    }

    vec3 node_T;

    node_T.x = transform[0] * nodes.at(number).x + transform[1] * nodes.at(number).y + transform[2] * nodes.at(number).z + transform[3];
    node_T.y = transform[4] * nodes.at(number).x + transform[5] * nodes.at(number).y + transform[6] * nodes.at(number).z + transform[7];
    node_T.z = transform[8] * nodes.at(number).x + transform[9] * nodes.at(number).y + transform[10] * nodes.at(number).z + transform[11];

    return node_T;
}

std::vector<float> Cone::getNodeRadii() const{
    std::vector<float> radii_T;
    radii_T.resize(2);
    for( int i=0; i<2; i++ ){

        vec3 n0(0,0,0), nx(radii.at(i),0,0);
        vec3 n0_T, nx_T;

        vecmult(transform,n0,n0_T);
        vecmult(transform,nx,nx_T);

        radii_T.at(i) = (nx_T-n0_T).magnitude();

    }
    return radii_T;
}

float Cone::getNodeRadius( int number ) const{
    if( number<0 || number>1 ){
        throw( std::runtime_error("ERROR (Cone::getNodeRadius): node number must be 0 or 1."));
    }

    vec3 n0(0,0,0), nx(radii.at(number),0,0);
    vec3 n0_T, nx_T;

    vecmult(transform,n0,n0_T);
    vecmult(transform,nx,nx_T);

    return (nx_T-n0_T).magnitude();
}

uint Cone::getSubdivisionCount() const{
    return subdiv;
}

helios::vec3 Cone::getAxisUnitVector() const{

    std::vector<vec3> nodes_T;
    nodes_T.resize( 2 );

    for( uint i=0; i<2; i++ ){
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    helios::vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z );
    float length = powf(powf(axis_unit_vector.x,2) + powf(axis_unit_vector.y,2) + powf(axis_unit_vector.z,2),0.5);
    axis_unit_vector = axis_unit_vector / length;

    return axis_unit_vector;
}

float Cone::getLength() const{

    std::vector<vec3> nodes_T;
    nodes_T.resize( 2);

    for( uint i=0; i<2; i++ ){
        nodes_T.at(i).x = transform[0] * nodes.at(i).x + transform[1] * nodes.at(i).y + transform[2] * nodes.at(i).z + transform[3];
        nodes_T.at(i).y = transform[4] * nodes.at(i).x + transform[5] * nodes.at(i).y + transform[6] * nodes.at(i).z + transform[7];
        nodes_T.at(i).z = transform[8] * nodes.at(i).x + transform[9] * nodes.at(i).y + transform[10] * nodes.at(i).z + transform[11];
    }

    float length = powf(powf(nodes_T.at(1).x - nodes_T.at(0).x, 2) + powf(nodes_T.at(1).y - nodes_T.at(0).y, 2) + powf(nodes_T.at(1).z - nodes_T.at(0).z, 2), 0.5);
    return length;
}

void Cone::scaleLength( float S ){

    //get the nodes and radii of the nodes with transformation matrix applied
    std::vector<helios::vec3> nodes_T = context->getConeObjectPointer(OID)->getNodes();
    std::vector<float> radii_T = context->getConeObjectPointer(OID)->getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z );
    float length = powf(powf(axis_unit_vector.x,2) + powf(axis_unit_vector.y,2) + powf(axis_unit_vector.z,2),0.5);
    axis_unit_vector = axis_unit_vector / length;

    //translate node 0 back to origin
    context->getConeObjectPointer(OID)->translate(-1.0*nodes_T.at(0));

    //rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0,0,1);
    //get the axis about which to rotate
    vec3 ra = cross( z_axis, axis_unit_vector);
    //get the angle to rotate
    float dot = axis_unit_vector.x*z_axis.x + axis_unit_vector.y*z_axis.y + axis_unit_vector.z*z_axis.z;
    float angle = acos_safe(dot);

    //only rotate if the cone is not alread aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if(angle != float(0.0)){
        context->getConeObjectPointer(OID)->rotate( -1*angle, ra );
    }

    // scale the cone in the z (length) dimension
    float T[16], T_prim[16];
    makeScaleMatrix( make_vec3(1,1,S), T);
    matmult(T,transform,transform);
    for( uint UUID : UUIDs){
        if( context->doesPrimitiveExist( UUID ) ){
            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);
        }
    }

    //rotate back
    if(angle != 0.0){
        context->getConeObjectPointer(OID)->rotate( angle, ra );
    }

    // //translate back
    context->getConeObjectPointer(OID)->translate(nodes_T.at(0));

}

void Cone::scaleGirth( float S ){

    //get the nodes and radii of the nodes with transformation matrix applied
    std::vector<helios::vec3> nodes_T = context->getConeObjectPointer(OID)->getNodes();
    std::vector<float> radii_T = context->getConeObjectPointer(OID)->getNodeRadii();

    // calculate the transformed axis unit vector of the cone
    vec3 axis_unit_vector = helios::make_vec3(nodes_T.at(1).x - nodes_T.at(0).x, nodes_T.at(1).y - nodes_T.at(0).y, nodes_T.at(1).z - nodes_T.at(0).z );
    float length = powf(powf(axis_unit_vector.x,2) + powf(axis_unit_vector.y,2) + powf(axis_unit_vector.z,2),0.5);
    axis_unit_vector = axis_unit_vector / length;

    //translate node 0 back to origin
    context->getConeObjectPointer(OID)->translate(-1.0*nodes_T.at(0));

    //rotate the cone to align with z axis
    helios::vec3 z_axis = make_vec3(0,0,1);
    //get the axis about which to rotate
    vec3 ra = cross( z_axis, axis_unit_vector);
    //get the angle to rotate
    float dot = axis_unit_vector.x*z_axis.x + axis_unit_vector.y*z_axis.y + axis_unit_vector.z*z_axis.z;
    float angle = acos_safe(dot);
    //only rotate if the cone is not alread aligned with the z axis (i.e., angle is not zero. If zero, the axis of rotation is 0,0,0 and we end up with problems)
    if(angle != float(0.0)){
        context->getConeObjectPointer(OID)->rotate( -1*angle, ra );
    }

    // scale the cone in the z (length) dimension
    float T[16], T_prim[16];
    makeScaleMatrix( make_vec3(S,S,1), T);
    matmult(T,transform,transform);
    for( uint UUID : UUIDs){
        if( context->doesPrimitiveExist( UUID ) ){
            context->getPrimitivePointer( UUID )->getTransformationMatrix(T_prim);
            matmult(T,T_prim,T_prim);
            context->getPrimitivePointer( UUID )->setTransformationMatrix(T_prim);
        }
    }

    //rotate back
    if(angle != 0.0){
        context->getConeObjectPointer(OID)->rotate( angle, ra );
    }

    // //translate back
    context->getConeObjectPointer(OID)->translate(nodes_T.at(0));

}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius ){

    RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addSphereObject(Ndivs,center,radius,color);

}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color ){

    if( radius<=0.f ){
        throw( std::runtime_error("ERROR (addSphereObject): Radius of sphere must be positive."));
    }

    std::vector<uint> UUID;

    float theta;
    float dtheta=float(M_PI)/float(Ndivs);
    float dphi=2.0f*float(M_PI)/float(Ndivs);

    //bottom cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j+1)*dphi ) );

        UUID.push_back( addTriangle(v0,v1,v2,color) );

    }

    //top cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j+1)*dphi ) );

        UUID.push_back( addTriangle(v2,v1,v0,color) );

    }

    //middle
    for( int j=0; j<Ndivs; j++ ){
        for( int i=1; i<Ndivs-1; i++ ){

            vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j+1)*dphi ) );

            UUID.push_back( addTriangle(v0,v1,v2,color) );
            UUID.push_back( addTriangle(v0,v2,v3,color) );

        }
    }

    auto* sphere_new = (new Sphere( currentObjectID, UUID, Ndivs, this ));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(radius,radius,radius),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    sphere_new->setTransformationMatrix( transform );

    sphere_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = sphere_new;
    currentObjectID++;
    return currentObjectID-1;


}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const char* texturefile ){

    if( radius<=0.f ){
        throw( std::runtime_error("ERROR (addSphereObject): Radius of sphere must be positive."));
    }

    std::vector<uint> UUID;

    float theta;
    float dtheta=float(M_PI)/float(Ndivs);
    float dphi=2.0f*float(M_PI)/float(Ndivs);

    //bottom cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j)*dphi ) );

        vec3 n0 = v0-center;
        n0.normalize();
        vec3 n1 = v1-center;
        n1.normalize();
        vec3 n2 = v2-center;
        n2.normalize();

        vec2 uv0 = make_vec2( 1.f - atan2f( sinf((float(j)+0.5f)*dphi), -cosf((float(j)+0.5f)*dphi) )/float(2.f*M_PI) - 0.5f, 1.f - n0.z*0.5f - 0.5f );
        vec2 uv1 = make_vec2( 1.f - atan2f( n1.x, -n1.y )/float(2.f*M_PI) - 0.5f, 1.f - n1.z*0.5f - 0.5f );
        vec2 uv2 = make_vec2( 1.f - atan2f( n2.x, -n2.y )/float(2.f*M_PI) - 0.5f, 1.f - n2.z*0.5f - 0.5f );

        if( j==Ndivs-1 ){
            uv2.x = 1;
        }

        UUID.push_back( addTriangle(v0,v1,v2,texturefile,uv0,uv1,uv2) );

    }

    //top cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j)*dphi ) );

        vec3 n0 = v0-center;
        n0.normalize();
        vec3 n1 = v1-center;
        n1.normalize();
        vec3 n2 = v2-center;
        n2.normalize();

        vec2 uv0 = make_vec2( 1.f - atan2f( sin((float(j)+0.5f)*dphi), -cos((float(j)+0.5f)*dphi) )/float(2.f*M_PI) - 0.5f, 1.f - n0.z*0.5f - 0.5f );
        vec2 uv1 = make_vec2( 1.f - atan2f( n1.x, -n1.y )/float(2.f*M_PI) - 0.5f, 1.f - n1.z*0.5f - 0.5f );
        vec2 uv2 = make_vec2( 1.f - atan2f( n2.x, -n2.y )/float(2.f*M_PI) - 0.5f, 1.f - n2.z*0.5f - 0.5f );

        if( j==Ndivs-1 ){
            uv2.x = 1;
        }

        UUID.push_back( addTriangle(v0,v1,v2,texturefile,uv0,uv1,uv2) );

    }

    //middle
    for( int j=0; j<Ndivs; j++ ){
        for( int i=1; i<Ndivs-1; i++ ){

            vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j+1)*dphi ) );

            vec3 n0 = v0-center;
            n0.normalize();
            vec3 n1 = v1-center;
            n1.normalize();
            vec3 n2 = v2-center;
            n2.normalize();
            vec3 n3 = v3-center;
            n3.normalize();

            vec2 uv0 = make_vec2( 1.f - atan2f( n0.x, -n0.y )/float(2.f*M_PI) - 0.5f, 1.f - n0.z*0.5f - 0.5f );
            vec2 uv1 = make_vec2( 1.f - atan2f( n1.x, -n1.y )/float(2.f*M_PI) - 0.5f, 1.f - n1.z*0.5f - 0.5f );
            vec2 uv2 = make_vec2( 1.f - atan2f( n2.x, -n2.y )/float(2.f*M_PI) - 0.5f, 1.f - n2.z*0.5f - 0.5f );
            vec2 uv3 = make_vec2( 1.f - atan2f( n3.x, -n3.y )/float(2.f*M_PI) - 0.5f, 1.f - n3.z*0.5f - 0.5f );

            if( j==Ndivs-1 ){
                uv2.x = 1;
                uv3.x = 1;
            }

            UUID.push_back( addTriangle(v0,v1,v2,texturefile,uv0,uv1,uv2) );
            UUID.push_back( addTriangle(v0,v2,v3,texturefile,uv0,uv2,uv3) );

        }
    }

    auto* sphere_new = (new Sphere( currentObjectID, UUID, Ndivs, this ));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(radius,radius,radius),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    sphere_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = sphere_new;
    currentObjectID++;
    return currentObjectID-1;


}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const int2 &subdiv, const SphericalCoord &rotation) {

    RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addTileObject(center,size,rotation,subdiv,color);
}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color ){

    if( size.x==0 || size.y==0 ){
        throw( std::runtime_error("ERROR (addTileObject): Size of tile must be greater than 0."));
    }
    if( subdiv.x<1 || subdiv.y<1 ){
        throw( std::runtime_error("ERROR (addTileObject): Number of tile subdivisions must be greater than 0."));
    }

    std::vector<uint> UUID;

    vec2 subsize;
    subsize.x = size.x/float(subdiv.x);
    subsize.y = size.y/float(subdiv.y);

    vec3 subcenter;

    for( uint j=0; j<subdiv.y; j++ ){
        for( uint i=0; i<subdiv.x; i++ ){

            subcenter = make_vec3(-0.5f*size.x+(float(i)+0.5f)*subsize.x,-0.5f*size.y+(float(j)+0.5f)*subsize.y,0.f);

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

    auto* tile_new = (new Tile( currentObjectID, UUID, subdiv, this ));

    float T[16], S[16], R[16];

    float transform[16];
    tile_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(size.x,size.y,1.f),S);
    matmult(S,transform,transform);

    makeRotationMatrix( -rotation.elevation,"x",R);
    matmult(R,transform,transform);
    makeRotationMatrix( -rotation.azimuth,"z",R);
    matmult(R,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    tile_new->setTransformationMatrix( transform );

    tile_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tile_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char* texturefile ){

    if( size.x==0 || size.y==0 ){
        throw( std::runtime_error("ERROR (addTileObject): Size of tile must be greater than 0."));
    }
    if( subdiv.x<1 || subdiv.y<1 ){
        throw( std::runtime_error("ERROR (addTileObject): Number of tile subdivisions must be greater than 0."));
    }

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

            subcenter = make_vec3(-0.5f*size.x+(float(i)+0.5f)*subsize.x,-0.5f*size.y+(float(j)+0.5f)*subsize.y,0.f);

            uv.at(0) = make_vec2(1.f-float(i)*uv_sub.x,float(j)*uv_sub.y);
            uv.at(1) = make_vec2(1.f-float(i+1)*uv_sub.x,float(j)*uv_sub.y);
            uv.at(2) = make_vec2(1.f-float(i+1)*uv_sub.x,float(j+1)*uv_sub.y);
            uv.at(3) = make_vec2(1.f-float(i)*uv_sub.x,float(j+1)*uv_sub.y);

            float solid_fraction;
            if( texture->hasTransparencyChannel() ){
                int A = 0;
                int At = 0;

                int2 uv_min( floor(uv.at(1).x*(float(sz.x)-1.f)), floor(uv.at(1).y*(float(sz.y)-1.f)) );
                int2 uv_max( floor(uv.at(3).x*(float(sz.x)-1.f)), floor(uv.at(3).y*(float(sz.y)-1.f)) );

                assert( uv_min.x>=0 && uv_min.y>=0 && uv_max.x<sz.x && uv_max.y<sz.y );

                for( int jj=uv_min.y; jj<uv_max.y; jj++ ){
                    for( int ii=uv_min.x; ii<uv_max.x; ii++ ){
                        At += 1;
                        if( alpha->at(jj).at(ii) ){
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

            auto* patch_new = (new Patch( texture, uv, solid_fraction, currentUUID ));

            patch_new->setParentObjectID(0);

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

    auto* tile_new = (new Tile( currentObjectID, UUID, subdiv, this ));

    float T[16], S[16], R[16];

    float transform[16];
    tile_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(size.x,size.y,1.f),S);
    matmult(S,transform,transform);

    makeRotationMatrix( -rotation.elevation,"x",R);
    matmult(R,transform,transform);
    makeRotationMatrix( -rotation.azimuth,"z",R);
    matmult(R,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    tile_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tile_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addTubeObject(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius ){

    uint node_count = nodes.size();

    std::vector<RGBcolor> color;
    color.resize(node_count);

    for( uint i=0; i<node_count; i++ ){
        color.at(i) = make_RGBcolor(0.f,0.75f,0.f); //Default color is green
    }

    return addTubeObject(Ndivs,nodes,radius,color);

}

uint Context::addTubeObject(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color ){

    const uint node_count = nodes.size();

    if( node_count==0 ){
        throw( std::runtime_error("ERROR (Context::addTubeObject): Node and radius arrays are empty."));
    }else if( node_count!=radius.size() ){
        throw( std::runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `radius' arguments must agree."));
    }else if( node_count!=color.size() ){
        throw( std::runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `color' arguments must agree."));
    }

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
    std::vector<std::vector<vec3> > xyz, normal;
    xyz.resize(Ndivs+1);
    normal.resize(Ndivs+1);
    for( uint j=0; j<Ndivs+1; j++ ){
        xyz.at(j).resize(node_count);
        normal.at(j).resize(node_count);
    }
    vec3 nvec(0.1817f,0.6198f,0.7634f);//random vector to get things going

    for( int j=0; j<Ndivs+1; j++ ){
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
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

    auto* tube_new = (new Tube( currentObjectID, UUID, nodes, radius, Ndivs, this ));

    float T[16],  transform[16];
    tube_new->getTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tube_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addTubeObject(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char* texturefile ){

    const uint node_count = nodes.size();

    if( node_count==0 ){
        throw( std::runtime_error("ERROR (Context::addTubeObject): Node and radius arrays are empty."));
    }else if( node_count!=radius.size() ){
        throw( std::runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `radius' arguments must agree."));
    }

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
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
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
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

    auto* tube_new = (new Tube( currentObjectID, UUID, nodes, radius, Ndivs, this ));

    float T[16],  transform[16];
    tube_new->getTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = tube_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv ){

    RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addBoxObject(center,size,subdiv,color,false);
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color ){
    return addBoxObject(center,size,subdiv,color,false);
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile ){
    return addBoxObject(center,size,subdiv,texturefile,false);
}

uint Context::addBoxObject(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals ){

    if( size.x<=0 || size.y<=0 || size.z<=0 ){
        throw( std::runtime_error("ERROR (addBoxObject): Size of box must be positive."));
    }
    if( subdiv.x<1 || subdiv.y<1 || subdiv.z<1 ){
        throw( std::runtime_error("ERROR (addBoxObject): Number of box subdivisions must be positive."));
    }

    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x/float(subdiv.x);
    subsize.y = size.y/float(subdiv.y);
    subsize.z = size.z/float(subdiv.z);

    vec3 subcenter;
    uint objID;
    std::vector<uint> U, U_copy;

    if( reverse_normals ){ //normals point inward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5f*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5f*M_PI,0), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5f*M_PI,1.5f*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5f*M_PI,0.5f*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }else{ //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5f*M_PI,0), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5f*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5f*M_PI,0.5f*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5f*M_PI,1.5f*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }

    auto* box_new = (new Box( currentObjectID, UUID, subdiv, this ));

    float T[16], transform[16];
    box_new->getTransformationMatrix( transform );

    makeScaleMatrix(size,T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    box_new->setTransformationMatrix( transform );

    box_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = box_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addBoxObject(vec3 center, const vec3 &size, const int3 &subdiv, const char* texturefile, bool reverse_normals ){

    std::vector<uint> UUID;

    vec3 subsize;
    subsize.x = size.x/float(subdiv.x);
    subsize.y = size.y/float(subdiv.y);
    subsize.z = size.z/float(subdiv.z);

    vec3 subcenter;
    uint objID;
    std::vector<uint> U, U_copy;

    if( reverse_normals ){ //normals point inward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }else{ //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }

    auto* box_new = (new Box( currentObjectID, UUID, subdiv, this ));

    float T[16], transform[16];
    box_new->getTransformationMatrix( transform );

    makeScaleMatrix(size,T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    box_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = box_new;
    currentObjectID++;
    return currentObjectID-1;


}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size ){
    return addDiskObject(Ndivs,center,size,make_SphericalCoord(0,0),make_RGBAcolor(1,0,0,1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation ){
    return addDiskObject(Ndivs,center,size,rotation,make_RGBAcolor(1,0,0,1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color ){
    return addDiskObject(Ndivs,center,size,rotation,make_RGBAcolor(color,1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color ){

    std::vector<uint> UUID;
    UUID.resize(Ndivs);

    for( int i=0; i<Ndivs; i++ ){

        float dtheta = 2.f*float(M_PI)/float(Ndivs);

        UUID.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), color );
        getPrimitivePointer(UUID.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer(UUID.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer(UUID.at(i))->translate( center );

    }

    auto* disk_new = (new Disk( currentObjectID, UUID, Ndivs, this ));

    float T[16], transform[16];
    disk_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(size.x,size.y,1.f),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    disk_new->setTransformationMatrix( transform );

    disk_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = disk_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){

    std::vector<uint> UUID;
    UUID.resize(Ndivs);

    for( int i=0; i<Ndivs; i++ ){

        float dtheta = 2.f*float(M_PI)/float(Ndivs);

        UUID.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), texture_file, make_vec2(0.5,0.5), make_vec2(0.5f*(1.f+cosf(dtheta*float(i))),0.5f*(1.f+sinf(dtheta*float(i)))), make_vec2(0.5f*(1.f+cosf(dtheta*float(i+1))),0.5f*(1.f+sinf(dtheta*float(i+1))))  );
        getPrimitivePointer(UUID.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer(UUID.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer(UUID.at(i))->translate( center );

    }

    auto* disk_new = (new Disk( currentObjectID, UUID, Ndivs, this ));

    float T[16], transform[16];
    disk_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(size.x,size.y,1.f),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    disk_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = disk_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addPolymeshObject(const std::vector<uint> &UUIDs ){

    auto* polymesh_new = (new Polymesh( currentObjectID, UUIDs, this ));

    polymesh_new->setColor( getPrimitivePointer( UUIDs.front())->getColor() );

    float T[16], transform[16];
    polymesh_new->getTransformationMatrix( transform );

    makeTranslationMatrix( getPrimitivePointer( UUIDs.front())->getVertices().front(),T);
    matmult(T,transform,transform);

    polymesh_new->setTransformationMatrix( transform );

    for( uint UUID : UUIDs){
        getPrimitivePointer(UUID)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = polymesh_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1 ){

    RGBcolor color;
    color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green
    return addConeObject(Ndivs, node0, node1, radius0, radius1, color);
}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const RGBcolor &color ){

    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
    std::vector<std::vector<vec3> > xyz, normal;
    xyz.resize(Ndivs+1);
    normal.resize(Ndivs+1);
    for( uint j=0; j<Ndivs+1; j++ ){
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
    }
    vec3 nvec(0.1817f,0.6198f,0.7634f);//random vector to get things going

    for( int j=0; j<Ndivs+1; j++ ){
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
    }

    for( int i=0; i<2; i++ ){ //looping over cone segments

        if(i==0){
            vec.x=nodes[i+1].x-nodes[i].x;
            vec.y=nodes[i+1].y-nodes[i].y;
            vec.z=nodes[i+1].z-nodes[i].z;
        }else if(i==1){
            vec.x=nodes[i].x-nodes[i-1].x;
            vec.y=nodes[i].y-nodes[i-1].y;
            vec.z=nodes[i].z-nodes[i-1].z;
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
            normal[j][i].x=cfact[j]*radii[i]*nvec.x+sfact[j]*radii[i]*convec.x;
            normal[j][i].y=cfact[j]*radii[i]*nvec.y+sfact[j]*radii[i]*convec.y;
            normal[j][i].z=cfact[j]*radii[i]*nvec.z+sfact[j]*radii[i]*convec.z;

            xyz[j][i].x=nodes[i].x+normal[j][i].x;
            xyz[j][i].y=nodes[i].y+normal[j][i].y;
            xyz[j][i].z=nodes[i].z+normal[j][i].z;

            normal[j][i] = normal[j][i]/radii[i];
        }

    }

    vec3 v0, v1, v2;
    std::vector<uint> UUID;

    for( int i=0; i<2-1; i++ ){
        for( int j=0; j<Ndivs; j++ ){

            v0 = xyz[j][i];
            v1 = xyz[j+1][i+1];
            v2 = xyz[j+1][i];

            UUID.push_back(addTriangle( v0, v1, v2, color ));

            v0 = xyz[j][i];
            v1 = xyz[j][i+1];
            v2 = xyz[j+1][i+1];

            UUID.push_back(addTriangle( v0, v1, v2, color ));

        }
    }

    auto* cone_new = (new Cone( currentObjectID, UUID, node0, node1, radius0, radius1, Ndivs, this ));

    float T[16],  transform[16];
    cone_new->getTransformationMatrix( transform );

    makeTranslationMatrix(nodes.front(),T);
    matmult(T,transform,transform);

    cone_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = cone_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addConeObject(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char* texturefile ){

    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
    std::vector<std::vector<vec3> > xyz, normal;
    std::vector<std::vector<vec2> > uv;
    xyz.resize(Ndivs+1);
    normal.resize(Ndivs+1);
    uv.resize(Ndivs+1);
    for( uint j=0; j<Ndivs+1; j++ ){
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
        uv.at(j).resize(2);
    }
    vec3 nvec(0.f,1.f,0.f);

    for( int j=0; j<Ndivs+1; j++ ){
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
    }

    for( int i=0; i<2; i++ ){ //looping over cone segments

        if(i==0){
            vec.x=nodes[i+1].x-nodes[i].x;
            vec.y=nodes[i+1].y-nodes[i].y;
            vec.z=nodes[i+1].z-nodes[i].z;
        }else if(i==1){
            vec.x=nodes[i].x-nodes[i-1].x;
            vec.y=nodes[i].y-nodes[i-1].y;
            vec.z=nodes[i].z-nodes[i-1].z;
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
            normal[j][i].x=cfact[j]*radii[i]*nvec.x+sfact[j]*radii[i]*convec.x;
            normal[j][i].y=cfact[j]*radii[i]*nvec.y+sfact[j]*radii[i]*convec.y;
            normal[j][i].z=cfact[j]*radii[i]*nvec.z+sfact[j]*radii[i]*convec.z;

            xyz[j][i].x=nodes[i].x+normal[j][i].x;
            xyz[j][i].y=nodes[i].y+normal[j][i].y;
            xyz[j][i].z=nodes[i].z+normal[j][i].z;

            uv[j][i].x = float(i)/float(2-1);
            uv[j][i].y = float(j)/float(Ndivs);

            normal[j][i] = normal[j][i]/radii[i];
        }

    }

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    std::vector<uint> UUID;

    for( int i=0; i<2-1; i++ ){
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

    auto* cone_new = (new Cone( currentObjectID, UUID, node0, node1, radius0, radius1, Ndivs, this ));

    float T[16],  transform[16];
    cone_new->getTransformationMatrix( transform );

    makeTranslationMatrix(nodes.front(),T);
    matmult(T,transform,transform);

    cone_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = cone_new;
    currentObjectID++;
    return currentObjectID-1;

}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius ){

    RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addSphere(Ndivs,center,radius,color);

}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color ){

    std::vector<uint> UUID;

    float theta;
    float dtheta=M_PI/float(Ndivs);
    float dphi=2.0f*float(M_PI)/float(Ndivs);

    //bottom cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j)*dphi ) );

        UUID.push_back( addTriangle(v0,v1,v2,color) );

    }

    //top cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j)*dphi ) );

        UUID.push_back( addTriangle(v2,v1,v0,color) );

    }

    //middle
    for( int j=0; j<Ndivs; j++ ){
        for( int i=1; i<Ndivs-1; i++ ){

            vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j+1)*dphi ) );

            UUID.push_back( addTriangle(v0,v1,v2,color) );
            UUID.push_back( addTriangle(v0,v2,v3,color) );

        }
    }

    return UUID;


}

std::vector<uint> Context::addSphere(uint Ndivs, const vec3 &center, float radius, const char* texturefile ){

    std::vector<uint> UUID;

    float theta;
    float dtheta=M_PI/float(Ndivs);
    float dphi=2.0f*float(M_PI)/float(Ndivs);

    //bottom cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+dtheta, float(j+1)*dphi ) );

        vec3 n0 = v0-center;
        n0.normalize();
        vec3 n1 = v1-center;
        n1.normalize();
        vec3 n2 = v2-center;
        n2.normalize();

        vec2 uv0 = make_vec2( 1.f - atan2f( sin((float(j)+0.5f)*dphi), -cos((float(j)+0.5f)*dphi) )/float(2.f*M_PI) - 0.5f, 1.f - n0.z*0.5f - 0.5f );
        vec2 uv1 = make_vec2( 1.f - atan2f( n1.x, -n1.y )/float(2.f*M_PI) - 0.5f, 1.f - n1.z*0.5f - 0.5f );
        vec2 uv2 = make_vec2( 1.f - atan2f( n2.x, -n2.y )/float(2.f*M_PI) - 0.5f, 1.f - n2.z*0.5f - 0.5f );

        if( j==Ndivs-1 ){
            uv2.x = 1;
        }

        UUID.push_back( addTriangle(v0,v1,v2,texturefile,uv0,uv1,uv2) );

    }

    //top cap
    for( int j=0; j<Ndivs; j++ ){

        vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI), 0 ) );
        vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, 0.5f*float(M_PI)-dtheta, float(j)*dphi ) );

        vec3 n0 = v0-center;
        n0.normalize();
        vec3 n1 = v1-center;
        n1.normalize();
        vec3 n2 = v2-center;
        n2.normalize();

        vec2 uv0 = make_vec2( 1.f - atan2f( sinf((float(j)+0.5f)*dphi), -cosf((float(j)+0.5f)*dphi) )/float(2.f*M_PI) - 0.5f, 1.f - n0.z*0.5f - 0.5f );
        vec2 uv1 = make_vec2( 1.f - atan2f( n1.x, -n1.y )/float(2.f*M_PI) - 0.5f, 1.f - n1.z*0.5f - 0.5f );
        vec2 uv2 = make_vec2( 1.f - atan2f( n2.x, -n2.y )/float(2.f*M_PI) - 0.5f, 1.f - n2.z*0.5f - 0.5f );

        if( j==Ndivs-1 ){
            uv2.x = 1;
        }

        UUID.push_back( addTriangle(v0,v1,v2,texturefile,uv0,uv1,uv2) );

    }

    //middle
    for( int j=0; j<Ndivs; j++ ){
        for( int i=1; i<Ndivs-1; i++ ){

            vec3 v0 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v2 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + sphere2cart( make_SphericalCoord(radius, -0.5f*float(M_PI)+float(i)*dtheta, float(j+1)*dphi ) );

            vec3 n0 = v0-center;
            n0.normalize();
            vec3 n1 = v1-center;
            n1.normalize();
            vec3 n2 = v2-center;
            n2.normalize();
            vec3 n3 = v3-center;
            n3.normalize();

            vec2 uv0 = make_vec2( 1.f - atan2f( n0.x, -n0.y )/float(2.f*M_PI) - 0.5f, 1.f - n0.z*0.5f - 0.5f );
            vec2 uv1 = make_vec2( 1.f - atan2f( n1.x, -n1.y )/float(2.f*M_PI) - 0.5f, 1.f - n1.z*0.5f - 0.5f );
            vec2 uv2 = make_vec2( 1.f - atan2f( n2.x, -n2.y )/float(2.f*M_PI) - 0.5f, 1.f - n2.z*0.5f - 0.5f );
            vec2 uv3 = make_vec2( 1.f - atan2f( n3.x, -n3.y )/float(2.f*M_PI) - 0.5f, 1.f - n3.z*0.5f - 0.5f );

            if( j==Ndivs-1 ){
                uv2.x = 1;
                uv3.x = 1;
            }

            UUID.push_back( addTriangle(v0,v1,v2,texturefile,uv0,uv1,uv2) );
            UUID.push_back( addTriangle(v0,v2,v3,texturefile,uv0,uv2,uv3) );

        }
    }

    return UUID;


}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv ){

    RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addTile(center,size,rotation,subdiv,color);
}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const RGBcolor &color ){

    std::vector<uint> UUID;

    vec2 subsize;
    subsize.x = size.x/float(subdiv.x);
    subsize.y = size.y/float(subdiv.y);

    vec3 subcenter;

    UUID.resize( subdiv.x*subdiv.y );

    size_t t = 0;
    for( uint j=0; j<subdiv.y; j++ ){
        for( uint i=0; i<subdiv.x; i++ ){

            subcenter = make_vec3(-0.5f*size.x+(float(i)+0.5f)*subsize.x,-0.5f*size.y+(float(j)+0.5f)*subsize.y,0);

            UUID[t] = addPatch( subcenter, subsize, make_SphericalCoord(0,0), color );

            if( rotation.elevation!=0.f ){
                getPrimitivePointer( UUID[t] )->rotate( -rotation.elevation, "x" );
            }
            if( rotation.azimuth!=0.f ){
                getPrimitivePointer( UUID[t] )->rotate( -rotation.azimuth, "z" );
            }
            getPrimitivePointer( UUID[t] )->translate( center );

            t++;

        }
    }

    return UUID;

}

std::vector<uint> Context::addTile(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv, const char* texturefile ){

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
    std::vector<std::vector<bool> > alpha;
    int2 sz;
    if( texture->hasTransparencyChannel() ){
        alpha = *texture->getTransparencyData();
        sz = texture->getSize();
    }

    for( uint j=0; j<subdiv.y; j++ ){
        for( uint i=0; i<subdiv.x; i++ ){

            subcenter = make_vec3(-0.5f*size.x+(float(i)+0.5f)*subsize.x,-0.5f*size.y+(float(j)+0.5f)*subsize.y,0);

            uv[0] = make_vec2(1.f-float(i)*uv_sub.x,float(j)*uv_sub.y);
            uv[1] = make_vec2(1.f-float(i+1)*uv_sub.x,float(j)*uv_sub.y);
            uv[2] = make_vec2(1.f-float(i+1)*uv_sub.x,float(j+1)*uv_sub.y);
            uv[3] = make_vec2(1.f-float(i)*uv_sub.x,float(j+1)*uv_sub.y);

            float solid_fraction;
            if( texture->hasTransparencyChannel() ){
                int A = 0;
                int At = 0;

                int2 uv_min( floor(uv[1].x*float(sz.x-1)), floor(uv[1].y*float(sz.y-1)) );
                int2 uv_max( floor(uv[3].x*float(sz.x-1)), floor(uv[3].y*float(sz.y-1)) );

                assert( uv_min.x>=0 && uv_min.y>=0 && uv_max.x<sz.x && uv_max.y<sz.y );

                for( int jj=uv_min.y; jj<uv_max.y; jj++ ){
                    for( int ii=uv_min.x; ii<uv_max.x; ii++ ){
                        At += 1;
                        if( alpha[jj][ii] ){
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

            if( solid_fraction==0.f ){
                continue;
            }

            auto* patch_new = (new Patch( texture, uv, solid_fraction, currentUUID ));

            patch_new->setParentObjectID(0);

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

std::vector<uint> Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius ){

    uint node_count = nodes.size();

    std::vector<RGBcolor> color;
    color.resize(node_count);

    for( uint i=0; i<node_count; i++ ){
        color.at(i) = make_RGBcolor(0.f,0.75f,0.f); //Default color is green
    }

    return addTube(Ndivs,nodes,radius,color);

}

std::vector<uint> Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, std::vector<RGBcolor> &color ){

    const uint node_count = nodes.size();

    if( node_count==0 ){
        throw( std::runtime_error("ERROR (Context::addTube): Node and radius arrays are empty."));
    }else if( node_count!=radius.size() ){
        throw( std::runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree."));
    }else if( node_count!=color.size() ){
        throw( std::runtime_error("ERROR (Context::addTube): Size of `nodes' and `color' arguments must agree."));
    }

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
    std::vector<std::vector<vec3> > xyz, normal;
    xyz.resize(Ndivs+1);
    normal.resize(Ndivs+1);
    for( uint j=0; j<Ndivs+1; j++ ){
        xyz.at(j).resize(node_count);
        normal.at(j).resize(node_count);
    }
    vec3 nvec(0.1817f,0.6198f,0.7634f);//random vector to get things going

    for( int j=0; j<Ndivs+1; j++ ){
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
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

std::vector<uint> Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const char* texturefile ){

    const uint node_count = nodes.size();

    if( node_count==0 ){
        throw( std::runtime_error("ERROR (Context::addTube): Node and radius arrays are empty."));
    }else if( node_count!=radius.size() ){
        throw( std::runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree."));
    }

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
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
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
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

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv ){

    RGBcolor color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addBox(center,size,subdiv,color,false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color ){
    return addBox(center,size,subdiv,color,false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile ){
    return addBox(center,size,subdiv,texturefile,false);
}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const RGBcolor &color, bool reverse_normals ){

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
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }else{ //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), color );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }

    return UUID;

}

std::vector<uint> Context::addBox(const vec3 &center, const vec3 &size, const int3 &subdiv, const char* texturefile, bool reverse_normals ){

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
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }else{ //normals point outward

        // x-z faces (vertical)

        //right
        subcenter = center + make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,0), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //left
        subcenter = center - make_vec3(0,0.5f*size.y,0);
        U = addTile( subcenter, make_vec2(size.x,size.z), make_SphericalCoord(0.5*M_PI,M_PI), make_int2(subdiv.x,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // y-z faces (vertical)

        //front
        subcenter = center + make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,0.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //back
        subcenter = center - make_vec3(0.5f*size.x,0,0);
        U = addTile( subcenter, make_vec2(size.y,size.z), make_SphericalCoord(0.5*M_PI,1.5*M_PI), make_int2(subdiv.y,subdiv.z), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        // x-y faces (horizontal)

        //top
        subcenter = center + make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(0,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

        //bottom
        subcenter = center - make_vec3(0,0,0.5f*size.z);
        U = addTile( subcenter, make_vec2(size.x,size.y), make_SphericalCoord(M_PI,0), make_int2(subdiv.x,subdiv.y), texturefile );
        UUID.insert( UUID.end(), U.begin(), U.end() );

    }

    return UUID;

}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size ){
    return addDisk(Ndivs,center,size,make_SphericalCoord(0,0),make_RGBAcolor(1,0,0,1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation ){
    return addDisk(Ndivs,center,size,rotation,make_RGBAcolor(1,0,0,1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color ){
    return addDisk(Ndivs,center,size,rotation,make_RGBAcolor(color,1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color ){

    std::vector<uint> UUIDs;
    UUIDs.resize(Ndivs);

    for( int i=0; i<Ndivs; i++ ){

        float dtheta = 2.f*float(M_PI)/float(Ndivs);

        UUIDs.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), color );
        getPrimitivePointer(UUIDs.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer(UUIDs.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer(UUIDs.at(i))->translate( center );

    }

    return UUIDs;

}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){

    std::vector<uint> UUIDs;
    UUIDs.resize(Ndivs);

    for( int i=0; i<Ndivs; i++ ){

        float dtheta = 2.f*float(M_PI)/float(Ndivs);

        UUIDs.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), texture_file, make_vec2(0.5,0.5), make_vec2(0.5f*(1.f+cosf(dtheta*float(i))),0.5f*(1.f+sinf(dtheta*float(i)))), make_vec2(0.5f*(1.f+cosf(dtheta*float(i+1))),0.5f*(1.f+sinf(dtheta*float(i+1))))  );
        getPrimitivePointer(UUIDs.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer(UUIDs.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer(UUIDs.at(i))->translate( center );

    }

    return UUIDs;

}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1 ){

    RGBcolor color;
    color = make_RGBcolor(0.f,0.75f,0.f); //Default color is green

    return addCone(Ndivs, node0, node1, radius0, radius1, color);

}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, RGBcolor &color ){

    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
    std::vector<std::vector<vec3> > xyz, normal;
    xyz.resize(Ndivs+1);
    normal.resize(Ndivs+1);
    for( uint j=0; j<Ndivs+1; j++ ){
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
    }
    vec3 nvec(0.1817f,0.6198f,0.7634f);//random vector to get things going

    for( int j=0; j<Ndivs+1; j++ ){
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
    }

    for( int i=0; i<2; i++ ){ //looping over cone segments

        if(i==0){
            vec.x=nodes[i+1].x-nodes[i].x;
            vec.y=nodes[i+1].y-nodes[i].y;
            vec.z=nodes[i+1].z-nodes[i].z;
        }else if(i==1){
            vec.x=nodes[i].x-nodes[i-1].x;
            vec.y=nodes[i].y-nodes[i-1].y;
            vec.z=nodes[i].z-nodes[i-1].z;
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
            normal[j][i].x=cfact[j]*radii[i]*nvec.x+sfact[j]*radii[i]*convec.x;
            normal[j][i].y=cfact[j]*radii[i]*nvec.y+sfact[j]*radii[i]*convec.y;
            normal[j][i].z=cfact[j]*radii[i]*nvec.z+sfact[j]*radii[i]*convec.z;

            xyz[j][i].x=nodes[i].x+normal[j][i].x;
            xyz[j][i].y=nodes[i].y+normal[j][i].y;
            xyz[j][i].z=nodes[i].z+normal[j][i].z;

            normal[j][i] = normal[j][i]/radii[i];
        }

    }

    vec3 v0, v1, v2;
    std::vector<uint> UUID;

    for( int i=0; i<2-1; i++ ){
        for( int j=0; j<Ndivs; j++ ){

            v0 = xyz[j][i];
            v1 = xyz[j+1][i+1];
            v2 = xyz[j+1][i];

            UUID.push_back(addTriangle( v0, v1, v2, color ));

            v0 = xyz[j][i];
            v1 = xyz[j][i+1];
            v2 = xyz[j+1][i+1];

            UUID.push_back(addTriangle( v0, v1, v2, color ));


        }
    }

    return UUID;

}

std::vector<uint> Context::addCone(uint Ndivs, const vec3 &node0, const vec3 &node1, float radius0, float radius1, const char* texturefile ){

    std::vector<helios::vec3> nodes{node0, node1};
    std::vector<float> radii{radius0, radius1};

    vec3 vec, convec;
    std::vector<float> cfact(Ndivs+1);
    std::vector<float> sfact(Ndivs+1);
    std::vector<std::vector<vec3> > xyz, normal;
    std::vector<std::vector<vec2> > uv;
    xyz.resize(Ndivs+1);
    normal.resize(Ndivs+1);
    uv.resize(Ndivs+1);
    for( uint j=0; j<Ndivs+1; j++ ){
        xyz.at(j).resize(2);
        normal.at(j).resize(2);
        uv.at(j).resize(2);
    }
    vec3 nvec(0.f,1.f,0.f);

    for( int j=0; j<Ndivs+1; j++ ){
        cfact[j]=cosf(2.f*float(M_PI)*float(j)/float(Ndivs));
        sfact[j]=sinf(2.f*float(M_PI)*float(j)/float(Ndivs));
    }

    for( int i=0; i<2; i++ ){ //looping over cone segments

        if(i==0){
            vec.x=nodes[i+1].x-nodes[i].x;
            vec.y=nodes[i+1].y-nodes[i].y;
            vec.z=nodes[i+1].z-nodes[i].z;
        }else if(i==1){
            vec.x=nodes[i].x-nodes[i-1].x;
            vec.y=nodes[i].y-nodes[i-1].y;
            vec.z=nodes[i].z-nodes[i-1].z;
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
            normal[j][i].x=cfact[j]*radii[i]*nvec.x+sfact[j]*radii[i]*convec.x;
            normal[j][i].y=cfact[j]*radii[i]*nvec.y+sfact[j]*radii[i]*convec.y;
            normal[j][i].z=cfact[j]*radii[i]*nvec.z+sfact[j]*radii[i]*convec.z;

            xyz[j][i].x=nodes[i].x+normal[j][i].x;
            xyz[j][i].y=nodes[i].y+normal[j][i].y;
            xyz[j][i].z=nodes[i].z+normal[j][i].z;

            uv[j][i].x = float(i)/float(2-1);
            uv[j][i].y = float(j)/float(Ndivs);

            normal[j][i] = normal[j][i]/radii[i];
        }

    }

    vec3 v0, v1, v2;
    vec2 uv0, uv1, uv2;
    std::vector<uint> UUID;

    for( int i=0; i<2-1; i++ ){
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

    XMLfiles.emplace_back( filename );

    uint ID;
    std::vector<uint> UUID;

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    //load file
    pugi::xml_parse_result result = xmldoc.load_file(filename);

    //error checking
    if (!result){
        throw( std::runtime_error("failed.\n XML [" + std::string(filename) + "] parsed with errors, attr value: [" + xmldoc.child("node").attribute("attr").value() + "]\nError description: " + result.description() + "\nError offset: " + std::to_string(result.offset) + " (error at [..." + (filename + result.offset) + "]\n"));
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if( helios.empty() ){
        std::cout << "failed." << std::endl;
        throw( std::runtime_error("ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags."));
    }

    //-------------- TIME/DATE ---------------//

    for (pugi::xml_node p = helios.child("date"); p; p = p.next_sibling("date")){

        pugi::xml_node year_node = p.child("year");
        const char* year_str = year_node.child_value();
        int year = std::stoi( year_str );

        pugi::xml_node month_node = p.child("month");
        const char* month_str = month_node.child_value();
        int month = std::stoi( month_str );

        pugi::xml_node day_node = p.child("day");
        const char* day_str = day_node.child_value();
        int day = std::stoi( day_str );

        setDate( day, month, year );

    }

    for (pugi::xml_node p = helios.child("time"); p; p = p.next_sibling("time")){

        pugi::xml_node hour_node = p.child("hour");
        const char* hour_str = hour_node.child_value();
        int hour = std::stoi( hour_str );

        pugi::xml_node minute_node = p.child("minute");
        const char* minute_str = minute_node.child_value();
        int minute = std::stoi( minute_str );

        pugi::xml_node second_node = p.child("second");
        const char* second_str = second_node.child_value();
        int second = std::stoi( second_str );

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
        std::string texfile = deblank(texture_node.child_value());
        if( texfile.empty() ){
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
            if( uv.empty() ){
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
        std::string texfile = deblank(texture_node.child_value());
        if( texfile.empty() ){
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
                    throw( std::runtime_error("ERROR (loadXML): Only 1 vertex was given for triangle (requires 3)."));
                }else if( i==2) {
                    std::cout << "failed." << std::endl;
                    throw( std::runtime_error("ERROR (loadXML): Only 2 vertices were given for triangle (requires 3)."));
                }
            }

            vertices = vertices.next_sibling("vertex");

        }

        //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^//

        // * Add the Triangle * //
        if( strcmp(texture_file.c_str(),"none")==0 || uv.empty() ){
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
//            rotation = std::stof(rotation_str);
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
                    throw( std::runtime_error("ERROR (loadXML): Invalid hour of " + std::to_string(time_.x) + " given in timeseries. Hour must be positive and not greater than 23."));
                }else if( time_.y<0 || time_.y>59 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid minute of " + std::to_string(time_.y) + " given in timeseries. Minute must be positive and not greater than 59."));
                }else if( time_.z<0 || time_.z>59 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid second of " + std::to_string(time_.z) + " given in timeseries. Second must be positive and not greater than 59."));
                }
                time = make_Time(time_.x, time_.y,time_.z);
            }else{
                throw( std::runtime_error("ERROR (loadXML): No time was specified for timeseries datapoint."));
            }

            Date date;
            bool date_flag=false;

            pugi::xml_node date_node = d.child("date");
            const char* date_str = date_node.child_value();
            if( strlen(date_str)>0 ){
                int3 date_ = string2int3(date_str);
                if( date_.x<1 || date_.x>31 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid day of month " + std::to_string(date_.x) + " given in timeseries. Day must be greater than zero and not greater than 31."));
                }else if( date_.y<1 || date_.y>12 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid month of " + std::to_string(date_.y) + " given in timeseries. Month must be greater than zero and not greater than 12."));
                }else if( date_.z<1000 || date_.z>10000 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid year of " + std::to_string(date_.z) + " given in timeseries. Year should be in YYYY format."));
                }
                date = make_Date(date_.x, date_.y, date_.z );
                date_flag=true;
            }

            pugi::xml_node Jdate_node = d.child("dateJulian");
            const char* Jdate_str = Jdate_node.child_value();
            if( strlen(Jdate_str)>0 ){
                int2 date_ = string2int2(Jdate_str);
                if( date_.x<1 || date_.x>366 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid Julian day of year " + std::to_string(date_.x) + " given in timeseries. Julian day must be greater than zero and not greater than 366."));
                }else if( date_.y<1000 || date_.y>10000 ){
                    throw( std::runtime_error("ERROR (loadXML): Invalid year of " + std::to_string(date_.y) + " given in timeseries. Year should be in YYYY format."));
                }
                date = Julian2Calendar( date_.x, date_.y );
                date_flag=true;
            }

            if( !date_flag ){
                throw( std::runtime_error("ERROR (loadXML): No date was specified for timeseries datapoint."));
            }

            float value;
            pugi::xml_node value_node = d.child("value");
            const char* value_str = value_node.child_value();
            if( strlen(value_str)>0 ){
                value = std::stof(value_str);
            }else{
                throw( std::runtime_error("ERROR (loadXML): No value was specified for timeseries datapoint."));
            }

            addTimeseriesData(label,value,date,time);

        }

    }

    std::cout << "done." << std::endl;

    return UUID;

}

std::vector<std::string> Context::getLoadedXMLFiles() {
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

    for(auto primitive : primitives){

        uint p = primitive.first;

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
        if( !pdata.empty() ){
            for(const auto& label : pdata){
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
            for(float i : transform){
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;
            std::vector<vec2> uv = patch->getTextureUV();
            if( !uv.empty() ){
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
            for(float i : transform){
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            std::vector<vec2> uv = getTrianglePointer(p)->getTextureUV();
            if( !uv.empty() ){
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
            for(float i : transform){
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            outfile << "   </voxel>" << std::endl;

        }

    }

    // -- global data -- //

    for(const auto & iter : globaldata){
        std::string label = iter.first;
        GlobalData data = iter.second;
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

    for(const auto & iter : timeseries_data){

        std::string label = iter.first;

        std::vector<float> data = iter.second;
        std::vector<double> dateval = timeseries_datevalue.at(label);

        assert( data.size()==dateval.size() );

        outfile << "   <timeseries label=\"" << label << "\">" << std::endl;

        for( size_t i=0; i<data.size(); i++ ){

            Date a_date = queryTimeseriesDate( label.c_str(), i );
            Time a_time = queryTimeseriesTime( label.c_str(), i );

            outfile << "\t<datapoint>" << std::endl;

            outfile << "\t   <date>" << a_date.day << " " << a_date.month << " " << a_date.year << "</date>" << std::endl;

            outfile << "\t   <time>" << a_time.hour << " " << a_time.minute << " " << a_time.second << "</time>" << std::endl;

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

std::vector<uint> Context::loadPLY(const char* filename, const vec3 &origin, float height ){
    return loadPLY( filename, origin, height, make_SphericalCoord(0,0), make_RGBcolor(0,0,1) );
}

std::vector<uint> Context::loadPLY(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation ){
    return loadPLY( filename, origin, height, rotation, make_RGBcolor(0,0,1) );
}

std::vector<uint> Context::loadPLY(const char* filename, const vec3 &origin, float height, const RGBcolor &default_color ){
    return loadPLY( filename, origin, height, make_SphericalCoord(0,0), default_color );
}

std::vector<uint> Context::loadPLY(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color ){

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
        throw( std::runtime_error("Couldn't open " + std::string(filename) ));
    }

    //--- read header info -----//

    //first line should always be 'ply'
    inputPly>>line;
    if( strcmp("ply",line.c_str())!=0 ){
        throw( std::runtime_error("ERROR (loadPLY): " + std::string(filename) + " is not a PLY file."));
    }

    //read format
    inputPly>>line;
    if( strcmp("format",line.c_str())!=0 ){
        throw( std::runtime_error("ERROR (loadPLY): could not determine data format of " + std::string(filename) ));
    }

    inputPly>>line;
    if( strcmp("ascii",line.c_str())!=0 ){
        throw( std::runtime_error("ERROR (loadPLY): Only ASCII data types are supported."));
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

    for(auto & propertie : properties){
        if( strcmp(propertie.c_str(),"red")==0 ){
            ifColor = true;
        }
    }
    std::cout<< "forming " << faceCount << " triangles..." << std::flush;

    vertices.resize(vertexCount);
    colors.resize(vertexCount);
    faces.resize(faceCount);


    //--- read vertices ----//

    for( uint row=0; row<vertexCount; row++ ){

        for(auto & propertie : properties){
            if( strcmp(propertie.c_str(),"x")==0 ){ //Note: permuting x,y,z to match our coordinate system (z-vertical instead of y-vertical)
                inputPly >> vertices.at(row).y;
            }else if( strcmp(propertie.c_str(),"y")==0 ){
                inputPly >> vertices.at(row).z;
            }else if( strcmp(propertie.c_str(),"z")==0 ){
                inputPly >> vertices.at(row).x;
            }else if( strcmp(propertie.c_str(),"red")==0 ){
                inputPly >> colors.at(row).r;
                colors.at(row).r /= 255.f;
            }else if( strcmp(propertie.c_str(),"green")==0 ){
                inputPly >> colors.at(row).g;
                colors.at(row).g /= 255.f;
            }else if( strcmp(propertie.c_str(),"blue")==0 ){
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

    for(auto primitive : primitives){

        uint p = primitive.first;

        std::vector<vec3> vertices = getPrimitivePointer(p)->getVertices();
        PrimitiveType type = getPrimitivePointer(p)->getType();
        RGBcolor C = getPrimitivePointer(p)->getColor();
        C.scale(255.f);

        if( type==PRIMITIVE_TYPE_TRIANGLE ){

            faces.push_back( make_int3( (int)vertex_count, (int)vertex_count+1, (int)vertex_count+2 ) );
            for( int i=0; i<3; i++ ){
                verts.push_back( vertices.at(i) );
                colors.push_back( C );
                vertex_count ++;
            }

        }else if( type==PRIMITIVE_TYPE_PATCH ){

            faces.push_back( make_int3( (int)vertex_count, (int)vertex_count+1, (int)vertex_count+2 ) );
            faces.push_back( make_int3( (int)vertex_count, (int)vertex_count+2, (int)vertex_count+3 ) );
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

    for(auto & face : faces){
        PLYfile << "3 " << face.x << " " << face.y << " " << face.z << std::endl;
    }

    PLYfile.close();


}

std::vector<uint> Context::loadOBJ(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color ){
    return loadOBJ(filename,origin,height,rotation,default_color,"ZUP");
}

std::vector<uint> Context::loadOBJ(const char* filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, const char* upaxis ){

    std::cout << "Reading OBJ file " << filename << "..." << std::flush;

    if( strcmp(upaxis,"XUP") != 0 && strcmp(upaxis,"YUP") != 0 && strcmp(upaxis,"ZUP") != 0 ){
        throw(std::runtime_error("ERROR (loadOBJ): Up axis of " + std::string(upaxis) + " is not valid.  Should be one of 'XUP', 'YUP', or 'ZUP'."));
    }

    std::string line, prop;

    std::vector<vec3> vertices;
    std::vector<vec2> texture_uv;
    std::map<std::string,std::vector<std::vector<int> > > face_inds, texture_inds;

    std::map<std::string,std::string> material_textures;

    std::vector<uint> UUID;

    std::ifstream inputOBJ, inputMTL;
    inputOBJ.open(filename);

    if (!inputOBJ.is_open()) {
        throw(std::runtime_error("ERROR (loadOBJ): Couldn't open " + std::string(filename) ));
    }

    //determine the base file path for 'filename'
    std::string fstring = filename;
    std::string filebase;
    for( size_t i=fstring.size()-1; i>=0; i-- ){
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
            material_textures = loadMTL( filebase, material_file );

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
            std::string tmp, digitf, digitu;
            std::vector<int> f, u;
            while( stream.good() ){

                stream >> tmp;

                digitf="";
                int ic = 0;
                for(char i : tmp){
                    if( isdigit(i) ){
                        digitf.push_back( i );
                        ic++;
                    }else{
                        break;
                    }
                }

                digitu="";
                for( int i=ic+1; i<tmp.size(); i++ ){
                    if( isdigit(tmp[i]) ){
                        digitu.push_back( tmp[i] );
                    }else{
                        break;
                    }
                }

                if( !digitf.empty() && !digitu.empty() ){
                    f.push_back( std::stoi(digitf) );
                    u.push_back( std::stoi(digitu) );
                }

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

        std::string material = iter->first;
        std::string texture;
        if( material_textures.find(material)!=material_textures.end() ){
            texture = material_textures.at(material);
        }

        for( size_t i=0; i<face_inds.at(material).size(); i++ ){

            for( uint t=2; t<face_inds.at(material).at(i).size(); t++ ){

                RGBcolor color = default_color;

                vec3 v0 = vertices.at(face_inds.at(material).at(i).at(0)-1);
                vec3 v1 = vertices.at(face_inds.at(material).at(i).at(t-1)-1);
                vec3 v2 = vertices.at(face_inds.at(material).at(i).at(t)-1);

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
                if( !texture.empty() && !texture_inds.at(material).at(i).empty() ){//has texture

                    if( t<texture_inds.at(material).at(i).size() ){
                        int iuv0 = texture_inds.at(material).at(i).at(0)-1;
                        int iuv1 = texture_inds.at(material).at(i).at(t-1)-1;
                        int iuv2 = texture_inds.at(material).at(i).at(t)-1;

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

std::map<std::string, std::string> Context::loadMTL(const std::string &filebase, const std::string &material_file ){

    std::ifstream inputMTL;

    std::string file = material_file;

    //first look for mtl file using path given in obj file
    inputMTL.open(file.c_str());
    if( !inputMTL.is_open() ){
        //if that doesn't work, try looking in the same directry where obj file is located
        file = filebase+file;
        file.erase( remove( file.begin(), file.end(), ' ' ), file.end() );
        for( size_t i=file.size()-1; i>=0; i-- ){
            if( strcmp(&file.at(i),"l")==0 ){
                break;
            }else{
                file.erase(file.begin()+(int)i);
            }
        }
        if( file.empty() ){
            throw(std::runtime_error("ERROR (loadMTL): Material file does not have correct file extension (.mtl)."));
        }
        inputMTL.open( file.c_str() );
        if( !inputMTL.is_open() ){
            throw(std::runtime_error("ERROR (loadMTL): Material file " + std::string(file) + " given in .obj file cannot be found."));
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
                        int e = (int)tmp.size();
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
    // - it would make more sense to write patches  as quads rather than two triangles

    std::cout << "Writing OBJ file " << filename << "..." << std::flush;
    std::ofstream file;

    char objfilename[50];
    sprintf(objfilename, "%s.obj", filename);
    file.open(objfilename);

    file << "# Helios generated OBJ File" << std::endl;
    file << "# baileylab.ucdavis.edu/software/helios" << std::endl;
    file << "mtllib " << filename << ".mtl" << std::endl;
    std::vector < int3 > faces;
    std::vector < vec3 > verts;
    std::vector < vec2 > uv;
    std::vector < int3 > uv_inds;
    std::vector < std::string > texture_list;
    std::vector < RGBcolor > colors;
    size_t vertex_count = 1;  //OBJ files start indices at 1
    size_t uv_count = 1;

    for (auto primitive : primitives) {

        uint p = primitive.first;

        std::vector < vec3 > vertices = getPrimitivePointer(p)->getVertices();
        PrimitiveType type = getPrimitivePointer(p)->getType();
        RGBcolor C = getPrimitivePointer(p)->getColor();

        if (type == PRIMITIVE_TYPE_TRIANGLE) {

            faces.push_back(
                    make_int3( (int)vertex_count, (int)vertex_count + 1, (int)vertex_count + 2));
            colors.push_back(C);
            for (int i = 0; i < 3; i++) {
                verts.push_back(vertices.at(i));
                vertex_count++;
            }

            std::vector < vec2 > uv_v = getTrianglePointer(p)->getTextureUV();
            if (getTrianglePointer(p)->hasTexture()) {
                uv_inds.push_back(make_int3( (int)uv_count, (int)uv_count + 1, (int)uv_count + 2));
                texture_list.push_back(getTrianglePointer(p)->getTextureFile());
                for (int i = 0; i < 3; i++) {
                    uv.push_back( make_vec2(1-uv_v.at(i).x,uv_v.at(i).y));
                    uv_count++;
                }
            } else {
                texture_list.emplace_back("");
                uv_inds.push_back(make_int3(-1, -1, -1));
            }

        } else if (type == PRIMITIVE_TYPE_PATCH) {
            faces.push_back(
                    make_int3( (int)vertex_count, (int)vertex_count + 1, (int)vertex_count + 2));
            faces.push_back(
                    make_int3( (int)vertex_count, (int)vertex_count + 2, (int)vertex_count + 3));
            colors.push_back(C);
            colors.push_back(C);
            for (int i = 0; i < 4; i++) {
                verts.push_back(vertices.at(i));
                vertex_count++;
            }
            std::vector < vec2 > uv_v;
            std::string texturefile;
            uv_v = getPatchPointer(p)->getTextureUV();
            texturefile = getPatchPointer(p)->getTextureFile();

            if (getPatchPointer(p)->hasTexture()) {
                texture_list.push_back(texturefile);
                texture_list.push_back(texturefile);
                uv_inds.push_back(make_int3( (int)uv_count, (int)uv_count + 1, (int)uv_count + 2));
                uv_inds.push_back(make_int3( (int)uv_count, (int)uv_count + 2, (int)uv_count + 3));
                if (uv_v.empty()) {  //default (u,v)
                    uv.push_back(make_vec2(0, 1));
                    uv.push_back(make_vec2(1, 1));
                    uv.push_back(make_vec2(1, 0));
                    uv.push_back(make_vec2(0, 0));
                    uv_count += 4;
                } else {  //custom (u,v)
                    for (int i = 0; i < 4; i++) {
                        uv.push_back(uv_v.at(i));
                        uv_count++;
                    }
                }
            } else {
                texture_list.emplace_back("");
                texture_list.emplace_back("");
                uv_inds.push_back(make_int3(-1, -1, -1));
                uv_inds.push_back(make_int3(-1, -1, -1));
            }
        }
    }

    assert(uv_inds.size() == faces.size());
    assert(texture_list.size() == faces.size());
    assert(colors.size() == faces.size());

    for (auto & vert : verts) {
        file << "v " << vert.x << " " << vert.y << " "
             << vert.z << std::endl;
    }

    for (auto & v : uv) {
        file << "vt " << v.x << " " << v.y << std::endl;
    }

    std::string current_texture;
    int material_count = 1;
    std::vector < size_t > exsit_mtl_list;

    current_texture = texture_list.at(0);
    file << "usemtl material" << material_count << std::endl;
    material_count++;
    exsit_mtl_list.push_back(0);

    if (uv_inds.at(0).x < 0) {
        file << "f " << faces.at(0).x << " " << faces.at(0).y << " "
             << faces.at(0).z << std::endl;
    } else {
        //assert( uv_inds.at(f).x <= uv.size() && uv_inds.at(f).y <= uv.size() && uv_inds.at(f).z <= uv.size() );
        file << "f " << faces.at(0).x << "/" << uv_inds.at(0).x << " "
             << faces.at(0).y << "/" << uv_inds.at(0).y << " " << faces.at(0).z
             << "/" << uv_inds.at(0).z << std::endl;
    }

    for (size_t f = 1; f < faces.size(); f++) {

        if (current_texture != texture_list.at(f)) {
            bool mtl_exist_flag = false;
            size_t mtl_index = 0;
            size_t mtl_index_f = 0;
            for (size_t index = 0; index < exsit_mtl_list.size(); index++) {
                if (texture_list.at(f) == texture_list.at(exsit_mtl_list[index])) {
                    mtl_exist_flag = true;
                    mtl_index = index;
                    mtl_index_f = exsit_mtl_list[index];
                    break;
                }
            }

            if (mtl_exist_flag) {
                current_texture = texture_list.at(mtl_index_f);
                file << "usemtl material" << (mtl_index + 1) << std::endl;  //we plus 1 here as we index mtl from 1 instead of 0 in the file.
            } else {
                current_texture = texture_list.at(f);
                file << "usemtl material" << material_count << std::endl;
                material_count++;
                exsit_mtl_list.push_back(f);
            }
        }

        if (uv_inds.at(f).x < 0) {
            file << "f " << faces.at(f).x << " " << faces.at(f).y << " "
                 << faces.at(f).z << std::endl;
        } else {
            //assert( uv_inds.at(f).x <= uv.size() && uv_inds.at(f).y <= uv.size() && uv_inds.at(f).z <= uv.size() );
            file << "f " << faces.at(f).x << "/" << uv_inds.at(f).x << " "
                 << faces.at(f).y << "/" << uv_inds.at(f).y << " " << faces.at(f).z
                 << "/" << uv_inds.at(f).z << std::endl;
        }
    }
    file.close();

    char mtlfilename[50];
    sprintf(mtlfilename, "%s.mtl", filename);
    file.open(mtlfilename);

    current_texture = "";
    material_count = 1;
    RGBcolor current_color = make_RGBcolor(0.010203, 0.349302, 0.8372910);
    std::vector < size_t > exsit_mtl_list2;

    if (texture_list.at(0).empty()) {  //has no texture
        if (current_color.r != colors.at(0).r && current_color.g != colors.at(0).g
            && current_color.b != colors.at(0).b) {  //new color
            current_texture = texture_list.at(0);
            current_color = colors.at(0);
            file << "newmtl material" << material_count << std::endl;
            file << "Ka " << current_color.r << " " << current_color.g << " "
                 << current_color.b << std::endl;
            file << "Kd " << current_color.r << " " << current_color.g << " "
                 << current_color.b << std::endl;
            file << "Ks 0.0 0.0 0.0" << std::endl;
            file << "illum 2 " << std::endl;
        }
    }

    else {
        current_texture = texture_list.at(0);
        file << "newmtl material" << material_count << std::endl;
        file << "Ka 1.0 1.0 1.0" << std::endl;
        file << "Kd 1.0 1.0 1.0" << std::endl;
        file << "Ks 0.0 0.0 0.0" << std::endl;
        file << "illum 2 " << std::endl;
        file << "map_Ka " << current_texture << std::endl;
        file << "map_Kd " << current_texture << std::endl;
        file << "map_d " << current_texture << std::endl;
    }

    material_count++;
    exsit_mtl_list2.push_back(0);

    for (size_t f = 1; f < faces.size(); f++) {
        bool mtl_exist_flag = false;
        size_t mtl_index = 0;
        size_t mtl_index_f = 0;
        for (size_t index = 0; index < exsit_mtl_list2.size(); index++) {
            if (not (texture_list.at(f) != texture_list.at(exsit_mtl_list2.at(index)))) {
                mtl_exist_flag = true;
                mtl_index = index;
                mtl_index_f = exsit_mtl_list2[index];
                break;
            }
        }
        if (not mtl_exist_flag) {

            if (texture_list.at(f).empty()) {
                if (current_color.r != colors.at(f).r
                    && current_color.g != colors.at(f).g
                    && current_color.b != colors.at(f).b) {  //new color

                    current_texture = texture_list.at(f);
                    current_color = colors.at(f);
                    file << "newmtl material" << material_count << std::endl;
                    file << "Ka " << current_color.r << " " << current_color.g << " "
                         << current_color.b << std::endl;
                    file << "Kd " << current_color.r << " " << current_color.g << " "
                         << current_color.b << std::endl;
                    file << "Ks 0.0 0.0 0.0" << std::endl;
                    file << "illum 2 " << std::endl;
                    material_count++;
                }
            } else {
                current_texture = texture_list.at(f);
                file << "newmtl material" << material_count << std::endl;
                file << "Ka 1.0 1.0 1.0" << std::endl;
                file << "Kd 1.0 1.0 1.0" << std::endl;
                file << "Ks 0.0 0.0 0.0" << std::endl;
                file << "illum 2 " << std::endl;
                file << "map_Ka " << current_texture << std::endl;
                file << "map_Kd " << current_texture << std::endl;
                file << "map_d " << current_texture << std::endl;
                material_count++;
                exsit_mtl_list2.push_back(f);
            }

        }
    }
    file.close();
    std::cout << "done." << std::endl;
}



Context::~Context(){

    for(auto & primitive : primitives){
        Primitive* prim = getPrimitivePointer(primitive.first);
        delete prim;
    }

    for(auto & object : objects){
        CompoundObject* obj = getObjectPointer(object.first);
        delete obj;
    }

}
