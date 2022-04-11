/** \file "Context.cpp" Context declarations. 
    \author Brian Bailey

    Copyright (C) 2016-2022  Brian Bailey

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

void Context::addTexture( const char* texture_file ){
    if( textures.find(texture_file)==textures.end() ){//texture has not already been added
        Texture text( texture_file );
        textures[ texture_file ] = text;
    }
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

const std::vector<std::vector<bool> >* Texture::getTransparencyData() const{
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

bool Primitive::hasTexture() const{
    if( texturefile.empty() ){
        return false;
    }else{
        return true;
    }
}

std::string Primitive::getTextureFile() const{
    return texturefile;
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

float Primitive::getSolidFraction() const{
    return solid_fraction;
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

void Patch::rotate( float rot, const helios::vec3& origin, const helios::vec3& axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Patch::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float R[16];
    makeRotationMatrix(rot,origin,axis,R);
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

void Triangle::rotate( float rot, const helios::vec3& origin, const helios::vec3& axis ){

    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Triangle::rotate): Cannot rotate individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    float R[16];
    makeRotationMatrix(rot,origin,axis,R);
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

void Voxel::rotate( float rot, const helios::vec3& origin, const helios::vec3& axis ){
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
    }

    return primitive_data_types.at(label);

}

uint Primitive::getPrimitiveDataSize( const char* label ) const{

    if( !doesPrimitiveDataExist( label ) ){
        throw( std::runtime_error( "ERROR (getPrimitiveData): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID) ) );
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


Patch::Patch( const RGBAcolor& a_color, uint a_UUID ){

    makeIdentityMatrix( transform );

    color = a_color;
    assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    solid_fraction = 1.f;
    texturefile = "";
    texturecoloroverridden = false;

}

Patch::Patch( const char* a_texturefile, float a_solid_fraction, uint a_UUID ){

    makeIdentityMatrix( transform );

    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    texturefile = a_texturefile;
    solid_fraction = a_solid_fraction;
    texturecoloroverridden = false;

}

Patch::Patch( const char* a_texturefile, const std::vector<vec2>& a_uv, float a_solid_fraction, uint a_UUID ){

    makeIdentityMatrix( transform );

    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;

    texturefile = a_texturefile;
    uv = a_uv;
    solid_fraction = a_solid_fraction;
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

Triangle::Triangle(  const vec3& a_vertex0, const vec3& a_vertex1, const vec3& a_vertex2, const RGBAcolor& a_color, uint a_UUID ){

    makeTransformationMatrix(a_vertex0,a_vertex1,a_vertex2);
    color = a_color;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;
    texturefile = "";
    solid_fraction = 1.f;
    texturecoloroverridden = false;

}

Triangle::Triangle( const vec3& a_vertex0, const vec3& a_vertex1, const vec3& a_vertex2, const char* a_texturefile, const std::vector<vec2>& a_uv, float a_solid_fraction, uint a_UUID ){

    makeTransformationMatrix(a_vertex0,a_vertex1,a_vertex2);
    color = make_RGBAcolor(RGB::red,1);
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;

    texturefile = a_texturefile;
    uv = a_uv;
    solid_fraction = a_solid_fraction;
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

vec3 Triangle::getCenter() const{

//    Y[0] = make_vec3( 0.f, 0.f, 0.f);
//    Y[1] = make_vec3( 0.f, 1.f, 0.f);
//    Y[2] = make_vec3( 1.f/3.f, 1.f, 0.f);

    vec3 center0 = make_vec3(1.f/3.f,2.f/3.f,0.f);
    vec3 center;

    center.x = transform[0] * center0.x + transform[1] * center0.y + transform[2] * center0.z + transform[3];
    center.y = transform[4] * center0.x + transform[5] * center0.y + transform[6] * center0.z + transform[7];
    center.z = transform[8] * center0.x + transform[9] * center0.y + transform[10] * center0.z + transform[11];

    return center;

}

Voxel::Voxel( const RGBAcolor& a_color, uint a_UUID ){

    makeIdentityMatrix(transform);

    color = a_color;
    assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
    solid_fraction = 1.f;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_VOXEL;
    texturefile = "";
    texturecoloroverridden = false;

}

float Voxel::getVolume(){

    vec3 size = getSize();

    return size.x*size.y*size.z;
}

vec3 Voxel::getCenter() const{

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

//    if( patch_new->getArea()==0 ){
//        throw( std::runtime_error("ERROR (Context::addPatch): Patch has area of zero.") );
//    }

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

    addTexture( texture_file );

    auto* patch_new = (new Patch( texture_file, textures.at(texture_file).getSolidFraction(), currentUUID ));

//    if( patch_new->getArea()==0 ){
//        throw( std::runtime_error("ERROR (Context::addPatch): Patch has area of zero.") );
//    }

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

uint Context::addPatch( const vec3& center, const vec2& size, const SphericalCoord& rotation, const char* texture_file, const helios::vec2& uv_center, const helios::vec2& uv_size ){

    if( size.x==0 || size.y==0 ){
        throw( std::runtime_error("ERROR (addPatch): Size of patch must be greater than 0.") );
    }

    if( uv_center.x-0.5*uv_size.x<-1e-3 || uv_center.y-0.5*uv_size.y<-1e-3 || uv_center.x+0.5*uv_size.x-1.f>1e-3 || uv_center.y+0.5*uv_size.y-1.f>1e-3 ){
        throw( std::runtime_error("ERROR (addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1.") );
    }

    addTexture( texture_file );

    std::vector<helios::vec2> uv;
    uv.resize(4);
    uv.at(0) = uv_center+make_vec2(-0.5f*uv_size.x,-0.5f*uv_size.y);
    uv.at(1) = uv_center+make_vec2(+0.5f*uv_size.x,-0.5f*uv_size.y);
    uv.at(2) =  uv_center+make_vec2(+0.5f*uv_size.x,+0.5f*uv_size.y);
    uv.at(3) =  uv_center+make_vec2(-0.5f*uv_size.x,+0.5f*uv_size.y);

    float solid_fraction;
    if( textures.at(texture_file).hasTransparencyChannel() ){
        const std::vector<std::vector<bool> >* alpha = textures.at(texture_file).getTransparencyData();
        int A = 0;
        int At = 0;
        int2 sz = textures.at(texture_file).getSize();
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
    auto* patch_new = (new Patch( texture_file, uv, solid_fraction, currentUUID ));

//    if( patch_new->getArea()==0 ){
//        throw( std::runtime_error("ERROR (Context::addPatch): Patch has area of zero.") );
//    }

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

//    if( tri_new->getArea()==0 ){
//        throw( std::runtime_error("ERROR (Context::addTriangle): Triangle has area of zero.") );
//    }

    tri_new->setParentObjectID(0);
    primitives[currentUUID] = tri_new;
    markGeometryDirty();
    currentUUID++;
    return currentUUID-1;
}

uint Context::addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texture_file, const helios::vec2& uv0, const helios::vec2& uv1, const helios::vec2& uv2 ){

    addTexture( texture_file );

    std::vector<helios::vec2> uv;
    uv.resize(3);
    uv.at(0) = uv0;
    uv.at(1) = uv1;
    uv.at(2) = uv2;

    float solid_fraction;
    if( textures.at(texture_file).hasTransparencyChannel() ){
        const std::vector<std::vector<bool> >* alpha = textures.at(texture_file).getTransparencyData();
        int2 sz = textures.at(texture_file).getSize();
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

    auto* tri_new = (new Triangle( vertex0, vertex1, vertex2, texture_file, uv, solid_fraction, currentUUID ));

//    if( tri_new->getArea()==0 ){
//        throw( std::runtime_error("ERROR (Context::addTriangle): Triangle has area of zero.") );
//    }

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

    if( size.x*size.y*size.z==0 ){
        throw( std::runtime_error("ERROR (Context::addVoxel): Voxel has size of zero.") );
    }

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
    getPrimitivePointer_private(UUID)->translate(shift);
}

void Context::translatePrimitive( const std::vector<uint>& UUIDs, const vec3& shift ){
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->translate(shift);
    }
}

void Context::rotatePrimitive(uint UUID, float rot, const char* axis ){
    getPrimitivePointer_private(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive( const std::vector<uint>& UUIDs, float rot, const char* axis ){
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->rotate(rot,axis);
    }
}

void Context::rotatePrimitive(uint UUID, float rot, const helios::vec3& axis ){
    getPrimitivePointer_private(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const vec3 &axis ){
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->rotate(rot,axis);
    }
}

void Context::rotatePrimitive( uint UUID, float rot, const helios::vec3& origin, const helios::vec3& axis ){
    getPrimitivePointer_private(UUID)->rotate(rot,origin,axis);
}

void Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const helios::vec3& origin, const vec3 &axis ){
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->rotate(rot,origin,axis);
    }
}

void Context::scalePrimitive(uint UUID, const helios::vec3& S ){
    getPrimitivePointer_private(UUID)->scale(S);
}

void Context::scalePrimitive( const std::vector<uint>& UUIDs, const helios::vec3& S ){
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->scale(S);
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

    if( prim->getParentObjectID()!=0 ){//primitive belongs to an object

        uint ObjID = prim->getParentObjectID();
        if( doesObjectExist(ObjID) ) {
            objects.at(ObjID)->deleteChildPrimitive(UUID);
        }
    }

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
        Patch* p = getPatchPointer_private(UUID);
        std::vector<vec2> uv = p->getTextureUV();
        vec2 size = p->getSize();
        float solid_fraction = p->getArea()/(size.x*size.y);
        Patch* patch_new;
        if( !p->hasTexture() ){
            patch_new = (new Patch( p->getColorRGBA(), currentUUID ));
        }else{
            std::string texture_file = p->getTextureFile();
            if( uv.size()==4 ){
                patch_new = (new Patch( texture_file.c_str(), uv, solid_fraction, currentUUID ));
            }else{
                patch_new = (new Patch( texture_file.c_str(), solid_fraction, currentUUID ));
            }
        }
        float transform[16];
        p->getTransformationMatrix(transform);
        patch_new->setTransformationMatrix(transform);
        patch_new->setParentObjectID(parentID);
        primitives[currentUUID] = patch_new;
    }else if( type==PRIMITIVE_TYPE_TRIANGLE ){
        Triangle* p = getTrianglePointer_private(UUID);
        std::vector<vec3> vertices = p->getVertices();
        std::vector<vec2> uv = p->getTextureUV();
        Triangle* tri_new;
        if( !p->hasTexture() ){
            tri_new = (new Triangle( vertices.at(0), vertices.at(1), vertices.at(2), p->getColorRGBA(), currentUUID ));
        }else{
            std::string texture_file = p->getTextureFile();
            float solid_fraction = p->getArea()/calculateTriangleArea( vertices.at(0), vertices.at(1), vertices.at(2) );
            tri_new = (new Triangle( vertices.at(0), vertices.at(1), vertices.at(2), texture_file.c_str(), uv, solid_fraction, currentUUID ));
        }
        float transform[16];
        p->getTransformationMatrix(transform);
        tri_new->setTransformationMatrix(transform);
        tri_new->setParentObjectID(parentID);
        primitives[currentUUID] = tri_new;
    }else if( type==PRIMITIVE_TYPE_VOXEL ){
        Voxel* p = getVoxelPointer_private(UUID);
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
        getPrimitivePointer_private(currentUUID)->overrideTextureColor();
    }

    markGeometryDirty();
    currentUUID++;
    return currentUUID-1;
}

void Context::copyPrimitiveData( uint UUID, uint oldUUID){
    //copy the primitive data
    std::vector<std::string> plabel = getPrimitivePointer_private(UUID)->listPrimitiveData();
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

Primitive* Context::getPrimitivePointer_private( uint UUID ) const{
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

Patch* Context::getPatchPointer_private(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        throw( std::runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " is not a patch.") );
    }
    return dynamic_cast<Patch*>(primitives.at(UUID));
}

helios::vec2 Context::getPatchSize( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPatchSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        throw( std::runtime_error("ERROR (getPatchSize): UUID of " + std::to_string(UUID) + " is not a patch.") );
    }
    return dynamic_cast<Patch*>(primitives.at(UUID))->getSize();
}

helios::vec3 Context::getPatchCenter( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getPatchCenter): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        throw( std::runtime_error("ERROR (getPatchCenter): UUID of " + std::to_string(UUID) + " is not a patch.") );
    }
    return dynamic_cast<Patch*>(primitives.at(UUID))->getCenter();
}

Triangle* Context::getTrianglePointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        throw( std::runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " is not a triangle.") );
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID));
}

Triangle* Context::getTrianglePointer_private(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        throw( std::runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " is not a triangle.") );
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID));
}

helios::vec3 Context::getTriangleVertex( uint UUID, uint number ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getTriangleVertex): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        throw( std::runtime_error("ERROR (getTriangleVertex): UUID of " + std::to_string(UUID) + " is not a triangle.") );
    }else if( number>2 ){
        throw( std::runtime_error("ERROR (getTriangleVertex): Vertex index must be one of 0, 1, or 2.") );
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID))->getVertex( number );
}

Voxel* Context::getVoxelPointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        throw( std::runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " is not a voxel.") );
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID));
}

Voxel* Context::getVoxelPointer_private(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        throw( std::runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " is not a voxel.") );
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID));
}

helios::vec3 Context::getVoxelSize( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getVoxelSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        throw( std::runtime_error("ERROR (getVoxelSize): UUID of " + std::to_string(UUID) + " is not a patch.") );
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID))->getSize();
}

helios::vec3 Context::getVoxelCenter( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        throw( std::runtime_error("ERROR (getVoxelCenter): UUID of " + std::to_string(UUID) + " does not exist in the Context.") );
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        throw( std::runtime_error("ERROR (getVoxelCenter): UUID of " + std::to_string(UUID) + " is not a patch.") );
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID))->getCenter();
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

        std::vector<vec3> verts = getPrimitivePointer_private(primitive.first)->getVertices();

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

        std::vector<vec3> verts = getPrimitivePointer_private( UUID )->getVertices();

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

        vertices = getPrimitivePointer_private(p)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.x<xbounds.x || vertex.x>xbounds.y ){
                uint parent_object_ID = getPrimitivePointer_private(p)->getParentObjectID();
                if(parent_object_ID == 0) {
                    deletePrimitive(p);
                    break;
                }
                deleteObject(parent_object_ID);
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

        vertices = getPrimitivePointer_private(p)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.y<ybounds.x || vertex.y>ybounds.y ){
                uint parent_object_ID = getPrimitivePointer_private(p)->getParentObjectID();
                if(parent_object_ID == 0) {
                    deletePrimitive(p);
                    break;
                }
                deleteObject(parent_object_ID);
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

        vertices = getPrimitivePointer_private(p)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.z<zbounds.x || vertex.z>zbounds.y ){
                uint parent_object_ID = getPrimitivePointer_private(p)->getParentObjectID();
                if(parent_object_ID == 0) {
                    deletePrimitive(p);
                    break;
                }
                deleteObject(parent_object_ID);
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

        vertices = getPrimitivePointer_private(UUID)->getVertices();

        for(auto & vertex : vertices){
            if( vertex.x<xbounds.x || vertex.x>xbounds.y || vertex.y<ybounds.x || vertex.y>ybounds.y || vertex.z<zbounds.x || vertex.z>zbounds.y ){
                uint parent_object_ID = getPrimitivePointer_private(UUID)->getParentObjectID();
                if(parent_object_ID == 0) {
                    deletePrimitive(UUID);
                    delete_count++;
                    break;
                }
                deleteObject(parent_object_ID);
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

    return UUIDs;

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
            area += context->getPrimitiveArea(UUID);
        }

    }

    return area;

}

void CompoundObject::setColor( const helios::RGBcolor& a_color ){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->setPrimitiveColor( UUID, a_color );
        }

    }

    color = make_RGBAcolor(a_color, 1.f);

}

void CompoundObject::setColor( const helios::RGBAcolor& a_color ){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->setPrimitiveColor( UUID, a_color );
        }

    }

    color = a_color;

}

RGBcolor CompoundObject::getColor()const {
    return make_RGBcolor( color.r, color.g, color.b );
}

RGBcolor CompoundObject::getColorRGB()const {
    return make_RGBcolor( color.r, color.g, color.b );
}

RGBAcolor CompoundObject::getColorRGBA()const {
    return color;
}

void CompoundObject::overrideTextureColor(){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->overridePrimitiveTextureColor( UUID );
        }

    }
}

void CompoundObject::useTextureColor(){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->usePrimitiveTextureColor( UUID );
        }

    }
}

bool CompoundObject::hasTexture() const{
    if( getTextureFile().empty() ){
        return false;
    }else{
        return true;
    }
}

std::string CompoundObject::getTextureFile() const{
    return texturefile;
}

void CompoundObject::translate( const helios::vec3& shift ){

    float T[16], T_prim[16];
    makeTranslationMatrix(shift,T);

    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);
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
                context->getPrimitiveTransformationMatrix( UUID,Rz_prim);
                matmult(Rz,Rz_prim,Rz_prim);
                context->setPrimitiveTransformationMatrix( UUID,Rz_prim);
            }
        }
    }else if( strcmp(axis,"y")==0 ){
        float Ry[16], Ry_prim[16];
        makeRotationMatrix(rot,"y",Ry);
        matmult(Ry,transform,transform);
        for( uint UUID : UUIDs){
            if( context->doesPrimitiveExist( UUID ) ){
                context->getPrimitiveTransformationMatrix( UUID,Ry_prim);
                matmult(Ry,Ry_prim,Ry_prim);
                context->setPrimitiveTransformationMatrix( UUID,Ry_prim);
            }
        }
    }else if( strcmp(axis,"x")==0 ){
        float Rx[16], Rx_prim[16];
        makeRotationMatrix(rot,"x",Rx);
        matmult(Rx,transform,transform);
        for( uint UUID : UUIDs){
            if( context->doesPrimitiveExist( UUID ) ){
                context->getPrimitiveTransformationMatrix( UUID,Rx_prim);
                matmult(Rx,Rx_prim,Rx_prim);
                context->setPrimitiveTransformationMatrix( UUID,Rx_prim);
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

            context->getPrimitiveTransformationMatrix( UUID,R_prim);
            matmult(R,R_prim,R_prim);
            context->setPrimitiveTransformationMatrix( UUID,R_prim);

        }

    }

}

void CompoundObject::rotate( float rot, const helios::vec3&  origin, const helios::vec3& axis ){

    float R[16], R_prim[16];
    makeRotationMatrix(rot,origin,axis,R);
    matmult(R,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitiveTransformationMatrix( UUID,R_prim);
            matmult(R,R_prim,R_prim);
            context->setPrimitiveTransformationMatrix( UUID,R_prim);

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

void CompoundObject::setPrimitiveUUIDs( const std::vector<uint> &a_UUIDs ){
    UUIDs = a_UUIDs;
}

void CompoundObject::deleteChildPrimitive( uint UUID ){
    auto it = find( UUIDs.begin(), UUIDs.end(), UUID );
    if( it!=UUIDs.end() ){
        std::iter_swap(it,UUIDs.end()-1);
        UUIDs.pop_back();
        primitivesarecomplete=false;
    }
}

void CompoundObject::deleteChildPrimitive( const std::vector<uint> &a_UUIDs ){
    for( uint UUID : a_UUIDs ){
        deleteChildPrimitive(UUID);
    }
}

bool CompoundObject::arePrimitivesComplete() const{
    return primitivesarecomplete;
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

bool Context::areObjectPrimitivesComplete( uint objID ) const{
   return getObjectPointer(objID)->arePrimitivesComplete();
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

uint Context::getObjectCount() const{
    return objects.size();
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
        getPrimitivePointer_private(p)->setParentObjectID( currentObjectID );
    }

    std::string texturefile = objects.at(ObjID)->getTextureFile();

    if( type==OBJECT_TYPE_TILE ){

        Tile* o = getTileObjectPointer( ObjID );

        int2 subdiv = o->getSubdivisionCount();

        auto* tile_new = (new Tile(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = tile_new;

    }else if( type==OBJECT_TYPE_SPHERE ){

        Sphere* o = getSphereObjectPointer( ObjID );

        uint subdiv = o->getSubdivisionCount();

        auto* sphere_new = (new Sphere(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = sphere_new;

    }else if( type==OBJECT_TYPE_TUBE ){

        Tube* o = getTubeObjectPointer( ObjID );

        std::vector<vec3> nodes = o->getNodes();
        std::vector<float> radius = o->getNodeRadii();
        std::vector<RGBcolor> colors = o->getNodeColors();
        uint subdiv = o->getSubdivisionCount();

        auto* tube_new = (new Tube(currentObjectID, UUIDs_copy, nodes, radius, colors, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = tube_new;

    }else if( type==OBJECT_TYPE_BOX ){

        Box* o = getBoxObjectPointer( ObjID );

        vec3 size = o->getSize();
        int3 subdiv = o->getSubdivisionCount();

        auto* box_new = (new Box(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = box_new;

    }else if( type==OBJECT_TYPE_DISK ){

        Disk* o = getDiskObjectPointer( ObjID );

        vec2 size = o->getSize();
        uint subdiv = o->getSubdivisionCount();

        auto* disk_new = (new Disk(currentObjectID, UUIDs_copy, subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = disk_new;

    }else if( type==OBJECT_TYPE_POLYMESH ){

        Polymesh* o = getPolymeshObjectPointer( ObjID );

        auto* polymesh_new = (new Polymesh(currentObjectID, UUIDs_copy, texturefile.c_str(), this));

        objects[currentObjectID] = polymesh_new;

    }else if( type==OBJECT_TYPE_CONE ){

        Cone* o = getConeObjectPointer( ObjID );

        std::vector<vec3> nodes = o->getNodes();
        std::vector<float> radius = o->getNodeRadii();
        uint subdiv = o->getSubdivisionCount();

        auto* cone_new = (new Cone(currentObjectID, UUIDs_copy, nodes.at(0), nodes.at(1), radius.at(0), radius.at(1),
                                   subdiv, texturefile.c_str(), this));

        objects[currentObjectID] = cone_new;

    }

    float T[16];
    getObjectPointer( ObjID )->getTransformationMatrix( T );

    getObjectPointer( currentObjectID )->setTransformationMatrix( T );


    markGeometryDirty();
    currentObjectID++;
    return currentObjectID-1;
}

std::vector<uint> Context::filterObjectsByData( const std::vector<uint> &IDs, const char* object_data, float threshold, const char* comparator) const{

    std::vector<uint> output_object_IDs;
    output_object_IDs.resize(IDs.size());
    uint passed_count=0;

    for(uint i=0;i<IDs.size();i++)
    {

        if( doesObjectDataExist(IDs.at(i), object_data) ){

            HeliosDataType type = getObjectDataType(IDs.at(i), object_data);
            if( type==HELIOS_TYPE_UINT ){
                uint R;
                getObjectData(IDs.at(i), object_data, R);
                if( strcmp(comparator,"<")==0 ){
                    if( R<threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,">")==0 ){
                    if( R>threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,"=")==0 ){
                    if( R==threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }

            }else if(type==HELIOS_TYPE_FLOAT){
                float R;
                getObjectData(IDs.at(i), object_data, R);

                if( strcmp(comparator,"<")==0 ){
                    if( R<threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,">")==0 ){
                    if( R>threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,"=")==0 ){
                    if( R==threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }

            }else if(type==HELIOS_TYPE_INT){
                int R;
                getObjectData(IDs.at(i), object_data, R);

                if( strcmp(comparator,"<")==0 ){
                    if( R<threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,">")==0 ){
                    if( R>threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,"=")==0 ){
                    if( R==threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }
            }else{
                std::cout << "WARNING: Object data not of type UINT, INT, or FLOAT. Filtering for other types not yet supported." << std::endl;
            }


        }
    }

    output_object_IDs.resize(passed_count);

    return output_object_IDs;

}

void Context::translateObject(uint ObjID, const vec3& shift ){
    getObjectPointer(ObjID)->translate(shift);
}

void Context::translateObject( const std::vector<uint>& ObjIDs, const vec3& shift ){
    for( uint ID : ObjIDs){
        getObjectPointer(ID)->translate(shift);
    }
}

void Context::rotateObject(uint ObjID, float rot, const char* axis ){
    getObjectPointer(ObjID)->rotate(rot,axis);
}

void Context::rotateObject( const std::vector<uint>& ObjIDs, float rot, const char* axis ){
    for( uint ID : ObjIDs){
        getObjectPointer(ID)->rotate(rot,axis);
    }
}

void Context::rotateObject(uint ObjID, float rot, const vec3& axis ){
    getObjectPointer(ObjID)->rotate(rot,axis);
}

void Context::rotateObject( const std::vector<uint>& ObjIDs, float rot, const vec3& axis ){
    for( uint ID : ObjIDs){
        getObjectPointer(ID)->rotate(rot,axis);
    }
}

void Context::rotateObject(uint ObjID, float rot, const vec3& origin, const vec3& axis ){
    getObjectPointer(ObjID)->rotate(rot,origin,axis);
}

void Context::rotateObject( const std::vector<uint>& ObjIDs, float rot, const vec3& origin, const vec3& axis ){
    for( uint ID : ObjIDs){
        getObjectPointer(ID)->rotate(rot,origin,axis);
    }
}

std::vector<uint> Context::getObjectPrimitiveUUIDs( const std::vector<uint> &ObjIDs ) const{

    std::vector<uint> output_UUIDs;

    for(uint i=0;i<ObjIDs.size();i++)
    {
        CompoundObject* pointer = getObjectPointer(ObjIDs.at(i));
        std::vector<uint> current_UUIDs = pointer->getPrimitiveUUIDs();
        output_UUIDs.insert( output_UUIDs.end(), current_UUIDs.begin(), current_UUIDs.end() );
    }
    return output_UUIDs;
}

std::vector<uint> Context::getObjectPrimitiveUUIDs( uint ObjID ) const{

    std::vector<uint> IDs{ObjID};
    std::vector<uint> output_UUIDs = getObjectPrimitiveUUIDs(IDs);
    return output_UUIDs;
}

helios::ObjectType Context::getObjectType( uint ObjID ) const{
    return getObjectPointer(ObjID)->getObjectType();
}

float Context::getTileObjectAreaRatio(const uint &ObjectID) const{

    if( getObjectPointer(ObjectID)->getObjectType() != OBJECT_TYPE_TILE )
    {
        std::cerr << "WARNING (getTileObjectAreaRatio): ObjectID " << ObjectID<< " is not a tile object. Skipping..." << std::endl;
        return 0.0;
    }else{
        int2 subdiv = getTileObjectPointer(ObjectID)->getSubdivisionCount();
        if(subdiv.x == int(1) && subdiv.y == int(1) )
        {
            return 1.0;
        }else{
            float area = getTileObjectPointer(ObjectID)->getArea();
            vec2 size = getTileObjectPointer(ObjectID)->getSize();

            float subpatch_area = (size.x/float(subdiv.x))*(size.y/float(subdiv.y));
            return area/subpatch_area;
        }

    }
}

std::vector<float> Context::getTileObjectAreaRatio(const std::vector<uint> &ObjectIDs) const {

    std::vector<float> AreaRatios(ObjectIDs.size());
    for( uint i=0; i<ObjectIDs.size(); i++ ){
        AreaRatios.at(i) = getTileObjectAreaRatio(ObjectIDs.at(i));
    }

    return AreaRatios;
}

void Context::setTileObjectSubdivisionCount(const std::vector<uint> &ObjectIDs, int2 new_subdiv)
{

    //check that all objects are Tile Objects, and get vector of texture files
    std::vector<uint> tile_ObjectIDs;
    std::vector<uint> textured_tile_ObjectIDs;


    std::vector<std::string> tex;
    // for(uint i=1;i<ObjectIDs.size();i++)
    for(uint OBJID : ObjectIDs)
    {

        //check if the object ID is a tile object and if it is add it the tile_ObjectIDs vector
        if( getObjectPointer(OBJID)->getObjectType() != OBJECT_TYPE_TILE )
        {
            std::cerr << "WARNING (setTileObjectSubdivisionCount): ObjectID " << OBJID << " is not a tile object. Skipping..." << std::endl;
        }else{
            //test if the tile is textured and push into two different vectors
            Patch* p = getPatchPointer_private(getObjectPointer(OBJID)->getPrimitiveUUIDs().at(0));
            if( !p->hasTexture() ){ //no texture
                tile_ObjectIDs.push_back(OBJID);
            }else{ //texture
                textured_tile_ObjectIDs.push_back(OBJID);
                tex.push_back(p->getTextureFile() );
            }
        }
    }

    //Here just call setSubdivisionCount directly for the non-textured tile objects
    for(unsigned int tile_ObjectID : tile_ObjectIDs){

        Tile* current_object_pointer = getTileObjectPointer(tile_ObjectID);
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);
        RGBcolor color = current_object_pointer->getColorRGB();

        std::vector<uint> UUIDs_new = addTile(center, size, rotation, new_subdiv, color );

        for( uint UUID : UUIDs_new ) {
            getPrimitivePointer_private(UUID)->setParentObjectID(tile_ObjectID);
        }

        current_object_pointer->setPrimitiveUUIDs(UUIDs_new);
        current_object_pointer->setSubdivisionCount(new_subdiv);
        deletePrimitive(UUIDs_old);
    }

    // get a vector of unique texture files that are represented in the input tile objects
    sort(tex.begin(), tex.end());
    std::vector<std::string>::iterator it;
    it = std::unique(tex.begin(),tex.end());
    tex.resize( std::distance(tex.begin(),it) );

    //create object templates for all the unique texture files
    std::vector<uint> object_templates;
    std::vector<std::vector<uint>> template_primitives;
    for(uint j=0;j<tex.size();j++)
    {
        //create a template object for the current texture
        uint object_template = addTileObject(make_vec3(0,0,0), make_vec2(1,1), nullrotation, new_subdiv, tex.at(j).c_str());
        object_templates.push_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.push_back(object_primitives);
    }

    //keep loop over objects on the outside, otherwise need to update textured_tile_ObjectIDs vector all the time
    //for each textured tile object
    for(uint i=0;i<textured_tile_ObjectIDs.size();i++)
    {
        //get info from current object
        Tile* current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
        std::string current_texture_file = current_object_pointer->getTextureFile();

        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);

        //for unique textures
        for(uint j=0;j<tex.size();j++)
        {
            //if the current tile object has the same texture file as the current unique texture file
            if(current_texture_file == tex.at(j))
            {
                //delete the original object primitives
                deletePrimitive(UUIDs_old);

                //copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));

                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);

                float IM[16];
                makeIdentityMatrix(IM);
                current_object_pointer->setTransformationMatrix(IM);

                current_object_pointer->scale(make_vec3(size.x, size.y, 1));

                //transform based on original object data
                if( rotation.elevation!=0 ){
                    current_object_pointer->rotate(-rotation.elevation , "x");
                }
                if( rotation.azimuth!=0 ){
                    current_object_pointer->rotate(rotation.azimuth, "z");
                }
                current_object_pointer->translate(center);

            }
        }
    }


    //delete the template (objects and primitives)
    deleteObject(object_templates);

}

void Context::setTileObjectSubdivisionCount(const std::vector<uint> &ObjectIDs, float area_ratio)
{

    //check that all objects are Tile Objects, and get vector of texture files
    std::vector<uint> tile_ObjectIDs;
    std::vector<uint> textured_tile_ObjectIDs;

    std::vector<std::string> tex;
    // for(uint i=1;i<ObjectIDs.size();i++)
    for(uint OBJID : ObjectIDs)
    {
        //check if the object ID is a tile object and if it is add it the tile_ObjectIDs vector
        if( getObjectPointer(OBJID)->getObjectType() != OBJECT_TYPE_TILE )
        {
            std::cerr << "WARNING (setTileObjectSubdivisionCount): ObjectID " << OBJID << " is not a tile object. Skipping..." << std::endl;
        }else{
            //test if the tile is textured and push into two different vectors
            Patch* p = getPatchPointer_private(getObjectPointer(OBJID)->getPrimitiveUUIDs().at(0));
            if( !p->hasTexture() ){ //no texture
                tile_ObjectIDs.push_back(OBJID);
            }else{ //texture
                textured_tile_ObjectIDs.push_back(OBJID);
                tex.push_back(p->getTextureFile() );
            }
        }
    }

    //Here just call setSubdivisionCount directly for the non-textured tile objects
    for(uint i=0;i<tile_ObjectIDs.size();i++)
    {
        Tile* current_object_pointer = getTileObjectPointer(tile_ObjectIDs.at(i));
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);
        RGBcolor color = current_object_pointer->getColorRGB();

        float tile_area = current_object_pointer->getArea();

        // subpatch dimensions needed to keep the correct ratio and have the solid fraction area = the input area
        float subpatch_dimension = sqrtf( tile_area / area_ratio);
        float subpatch_per_x = size.x / subpatch_dimension;
        float subpatch_per_y = size.y / subpatch_dimension;

        float option_1_AR = (tile_area / (size.x / ceil(subpatch_per_x) * size.y / floor(subpatch_per_y))) - area_ratio;
        float option_2_AR = (tile_area / (size.x / floor(subpatch_per_x) * size.y / ceil(subpatch_per_y))) - area_ratio;

        int2 new_subdiv;
        if((int)area_ratio == 1){
            new_subdiv = make_int2(1, 1);
        }else if(option_1_AR >= option_2_AR){
            new_subdiv = make_int2(ceil(subpatch_per_x), floor(subpatch_per_y));
        }else{
            new_subdiv = make_int2(floor(subpatch_per_x), ceil(subpatch_per_y));
        }


        std::vector<uint> UUIDs_new = addTile(center, size, rotation, new_subdiv, color );

        for( uint UUID : UUIDs_new ) {
            getPrimitivePointer_private(UUID)->setParentObjectID(tile_ObjectIDs.at(i));
        }

        current_object_pointer->setPrimitiveUUIDs(UUIDs_new);
        current_object_pointer->setSubdivisionCount(new_subdiv);
        deletePrimitive(UUIDs_old);
    }

    // get a vector of unique texture files that are represented in the input tile objects
    sort(tex.begin(), tex.end());
    std::vector<std::string>::iterator it;
    it = std::unique(tex.begin(),tex.end());
    tex.resize( std::distance(tex.begin(),it) );

    //create object templates for all the unique texture files
    // the assumption here is that all tile objects with the same texture have the same aspect ratio
    //if this is not true then the copying method won't work well because a new template will need to be created for each texture/aspect ratio combination

    std::vector<uint> object_templates;
    std::vector<std::vector<uint>> template_primitives;
    for(uint j=0;j<tex.size();j++)
    {
        //here we just want to get one tile object with the matching texture
        uint ii;
        for(uint i=0;i<textured_tile_ObjectIDs.size();i++)
        {
            //get info from current object
            Tile* current_object_pointer_b = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
            std::string current_texture_file_b = current_object_pointer_b->getTextureFile();
            //if the current tile object has the same texture file as the current unique texture file
            if(current_texture_file_b == tex.at(j))
            {
                ii=i;
                break;
            }
        }

        //get info from current object
        Tile* current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(ii));
        vec2 tile_size = current_object_pointer->getSize();
        float tile_area = current_object_pointer->getArea();

        // subpatch dimensions needed to keep the correct ratio and have the solid fraction area = the input area
        float subpatch_dimension = sqrtf( tile_area / area_ratio);
        float subpatch_per_x = tile_size.x / subpatch_dimension;
        float subpatch_per_y = tile_size.y / subpatch_dimension;

        float option_1_AR = (tile_area / (tile_size.x / ceil(subpatch_per_x) * tile_size.y / floor(subpatch_per_y))) - area_ratio;
        float option_2_AR = (tile_area / (tile_size.x / floor(subpatch_per_x) * tile_size.y / ceil(subpatch_per_y))) - area_ratio;

        int2 new_subdiv;
        if((int)area_ratio == 1){
            new_subdiv = make_int2(1, 1);
        }else if(option_1_AR >= option_2_AR){
            new_subdiv = make_int2(ceil(subpatch_per_x), floor(subpatch_per_y));
        }else{
            new_subdiv = make_int2(floor(subpatch_per_x), ceil(subpatch_per_y));
        }

        //create a template object for the current texture
        uint object_template = addTileObject(make_vec3(0,0,0), make_vec2(1,1), nullrotation, new_subdiv, tex.at(j).c_str());
        object_templates.push_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.push_back(object_primitives);
    }

    //keep loop over objects on the outside, otherwise need to update textured_tile_ObjectIDs vector all the time
    //for each textured tile object
    for(uint i=0;i<textured_tile_ObjectIDs.size();i++)
    {
        //get info from current object
        Tile* current_object_pointer = getTileObjectPointer(textured_tile_ObjectIDs.at(i));
        // std::string current_texture_file = getPrimitivePointer_private(current_object_pointer->getPrimitiveUUIDs().at(0))->getTextureFile();
        std::string current_texture_file = current_object_pointer->getTextureFile();
        // std::cout << "current_texture_file for ObjID " << textured_tile_ObjectIDs.at(i) << " = " << current_texture_file << std::endl;
        std::vector<uint> UUIDs_old = current_object_pointer->getPrimitiveUUIDs();

        vec2 size = current_object_pointer->getSize();
        vec3 center = current_object_pointer->getCenter();
        vec3 normal = current_object_pointer->getNormal();
        SphericalCoord rotation = cart2sphere(normal);

        //for unique textures
        for(uint j=0;j<tex.size();j++)
        {
            //if the current tile object has the same texture file as the current unique texture file
            if(current_texture_file == tex.at(j))
            {
                //delete the original object primitives
                deletePrimitive(UUIDs_old);

                //copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));

                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));

                int2 new_subdiv = getTileObjectPointer(object_templates.at(j))->getSubdivisionCount();
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);

                float IM[16];
                makeIdentityMatrix(IM);
                current_object_pointer->setTransformationMatrix(IM);

                current_object_pointer->scale(make_vec3(size.x, size.y, 1));

                if( rotation.elevation!=0 ){
                    current_object_pointer->rotate(-rotation.elevation , "x");
                }
                if( rotation.azimuth!=0 ){
                    current_object_pointer->rotate(rotation.azimuth, "z");
                }
                current_object_pointer->translate(center);

            }
        }
    }

    //delete the template (objects and primitives)
    deleteObject(object_templates);

}

Tile::Tile(uint a_OID, const std::vector<uint> &a_UUIDs, const int2 &a_subdiv, const char *a_texturefile, helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_TILE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
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

void Tile::setSubdivisionCount( const helios::int2 &a_subdiv ){
    subdiv = a_subdiv;
}

std::vector<helios::vec3> Tile::getVertices() const{

    std::vector<helios::vec3> vertices;
    vertices.resize(4);

    //subcenter = make_vec3(-0.5*size.x+(float(i)+0.5)*subsize.x,-0.5*size.y+(float(j)+0.5)*subsize.y,0);
    //Y[0] = make_vec3( -0.5f, -0.5f, 0.f);
    //Y[1] = make_vec3( 0.5f, -0.5f, 0.f);
    //Y[2] = make_vec3( 0.5f, 0.5f, 0.f);
    //Y[3] = make_vec3( -0.5f, 0.5f, 0.f);

    vertices.at(0) = context->getPrimitiveVertices( UUIDs.front() ).at(0);

    vertices.at(1) = context->getPrimitiveVertices( UUIDs.at( subdiv.x-1 ) ).at(1);

    vertices.at(2) = context->getPrimitiveVertices( UUIDs.at( subdiv.x*subdiv.y-1 ) ).at(2);

    vertices.at(3) = context->getPrimitiveVertices( UUIDs.at( subdiv.x*subdiv.y-subdiv.x ) ).at(3);

    return vertices;

}

vec3 Tile::getNormal() const{

    return context->getPrimitiveNormal( UUIDs.front() );

}

std::vector<helios::vec2> Tile::getTextureUV() const{

    std::vector<helios::vec2> uv{ make_vec2(0,0), make_vec2(1,0), make_vec2(1,1), make_vec2(0,1) };

    return uv;

}

void Tile::scale(const vec3 &S ){

    float T[16], T_prim[16];
    makeScaleMatrix( S, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs) {

        if (context->doesPrimitiveExist(UUID)) {

            context->getPrimitiveTransformationMatrix(UUID, T_prim);
            matmult(T, T_prim, T_prim);
            context->setPrimitiveTransformationMatrix(UUID, T_prim);

        }

    }

}

Sphere::Sphere(uint a_OID, const std::vector<uint> &a_UUIDs, uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_SPHERE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
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

void Sphere::setSubdivisionCount( uint a_subdiv ){
    subdiv = a_subdiv;
}

void Sphere::scale( float S ){

    float T[16], T_prim[16];
    makeScaleMatrix( make_vec3(S,S,S), T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);

        }

    }

}

Tube::Tube(uint a_OID, const std::vector<uint> &a_UUIDs, const std::vector<vec3> &a_nodes, const std::vector<float> &a_radius, const std::vector<helios::RGBcolor> &a_colors, uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_TUBE;
    UUIDs = a_UUIDs;
    nodes = a_nodes;
    radius = a_radius;
    colors = a_colors;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
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

std::vector<helios::RGBcolor> Tube::getNodeColors() const{
    return colors;
}

uint Tube::getSubdivisionCount() const{
    return subdiv;
}

void Tube::setSubdivisionCount( uint a_subdiv ){
    subdiv = a_subdiv;
}

void Tube::scale( float S ){

    float T[16], T_prim[16];
    makeScaleMatrix( make_vec3(S,S,S), T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);

        }

    }

}

Box::Box(uint a_OID, const std::vector<uint> &a_UUIDs, const int3 &a_subdiv, const char *a_texturefile,
         helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_BOX;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
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

void Box::setSubdivisionCount( const helios::int3 &a_subdiv ){
    subdiv = a_subdiv;
}

void Box::scale(const vec3 &S ){

    float T[16], T_prim[16];
    makeScaleMatrix( S, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);

        }

    }

}

Disk::Disk(uint a_OID, const std::vector<uint> &a_UUIDs, uint a_subdiv, const char *a_texturefile,
           helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_DISK;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
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

void Disk::setSubdivisionCount( uint a_subdiv ){
    subdiv = a_subdiv;
}

void Disk::scale(const vec3 &S ){

    float T[16], T_prim[16];
    makeScaleMatrix( S, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){

            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);

        }

    }

}

Polymesh::Polymesh(uint a_OID, const std::vector<uint> &a_UUIDs, const char *a_texturefile,
                   helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_POLYMESH;
    UUIDs = a_UUIDs;
    texturefile = a_texturefile;
    context = a_context;

}

Polymesh* Context::getPolymeshObjectPointer(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        throw( std::runtime_error("ERROR (getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context."));
    }
    return dynamic_cast<Polymesh*>(objects.at(ObjID));
}

Cone::Cone(uint a_OID, const std::vector<uint> &a_UUIDs, const vec3 &a_node0, const vec3 &a_node1, float a_radius0,
           float a_radius1, uint a_subdiv, const char *a_texturefile, helios::Context *a_context) {

    makeIdentityMatrix( transform );

    OID = a_OID;
    type = helios::OBJECT_TYPE_CONE;
    UUIDs = a_UUIDs;
    subdiv = a_subdiv;
    texturefile = a_texturefile;
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

void Cone::setSubdivisionCount( uint a_subdiv ){
    subdiv = a_subdiv;
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
            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);
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
            context->getPrimitiveTransformationMatrix( UUID,T_prim);
            matmult(T,T_prim,T_prim);
            context->setPrimitiveTransformationMatrix( UUID,T_prim);
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

    auto* sphere_new = (new Sphere(currentObjectID, UUID, Ndivs, "", this));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(radius,radius,radius),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    sphere_new->setTransformationMatrix( transform );

    sphere_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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

    auto* sphere_new = (new Sphere(currentObjectID, UUID, Ndivs, texturefile, this));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(radius,radius,radius),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    sphere_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = sphere_new;
    currentObjectID++;
    return currentObjectID-1;


}

uint Context::addTileObject(const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const int2 &subdiv) {

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
                getPrimitivePointer_private( UUID.back() )->rotate( -rotation.elevation, "x" );
            }
            if( rotation.azimuth!=0.f ){
                getPrimitivePointer_private( UUID.back() )->rotate( -rotation.azimuth, "z" );
            }
            getPrimitivePointer_private( UUID.back() )->translate( center );

        }
    }

    auto* tile_new = (new Tile(currentObjectID, UUID, subdiv, "", this));

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
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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

    addTexture( texturefile );
    const std::vector<std::vector<bool> >* alpha;
    int2 sz;
    if( textures.at(texturefile).hasTransparencyChannel() ){
        alpha = textures.at(texturefile).getTransparencyData();
        sz = textures.at(texturefile).getSize();
    }

    for( uint j=0; j<subdiv.y; j++ ){
        for( uint i=0; i<subdiv.x; i++ ){

            subcenter = make_vec3(-0.5f*size.x+(float(i)+0.5f)*subsize.x,-0.5f*size.y+(float(j)+0.5f)*subsize.y,0.f);

            uv.at(0) = make_vec2(float(i)*uv_sub.x,float(j)*uv_sub.y);
            uv.at(1) = make_vec2(float(i+1)*uv_sub.x,float(j)*uv_sub.y);
            uv.at(2) = make_vec2(float(i+1)*uv_sub.x,float(j+1)*uv_sub.y);
            uv.at(3) = make_vec2(float(i)*uv_sub.x,float(j+1)*uv_sub.y);

            float solid_fraction;
            if( textures.at(texturefile).hasTransparencyChannel() ){
                int A = 0;
                int At = 0;

                int2 uv_min( floor(uv.at(0).x*(float(sz.x)-1.f)), floor(uv.at(0).y*(float(sz.y)-1.f)) );
                int2 uv_max( floor(uv.at(2).x*(float(sz.x)-1.f)), floor(uv.at(2).y*(float(sz.y)-1.f)) );

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

            auto* patch_new = (new Patch( texturefile, uv, solid_fraction, currentUUID ));

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

    auto* tile_new = (new Tile(currentObjectID, UUID, subdiv, texturefile, this));

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
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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

    auto* tube_new = (new Tube(currentObjectID, UUID, nodes, radius, color, Ndivs, "", this));

    float T[16],  transform[16];
    tube_new->getTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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

    std::vector<RGBcolor> colors;

    auto* tube_new = (new Tube(currentObjectID, UUID, nodes, radius, colors, Ndivs, texturefile, this));

    float T[16],  transform[16];
    tube_new->getTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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

    auto* box_new = (new Box(currentObjectID, UUID, subdiv, "", this));

    float T[16], transform[16];
    box_new->getTransformationMatrix( transform );

    makeScaleMatrix(size,T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    box_new->setTransformationMatrix( transform );

    box_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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

    auto* box_new = (new Box(currentObjectID, UUID, subdiv, texturefile, this));

    float T[16], transform[16];
    box_new->getTransformationMatrix( transform );

    makeScaleMatrix(size,T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    box_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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
        getPrimitivePointer_private(UUID.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer_private(UUID.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer_private(UUID.at(i))->translate( center );

    }

    auto* disk_new = (new Disk(currentObjectID, UUID, Ndivs, "", this));

    float T[16], transform[16];
    disk_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(size.x,size.y,1.f),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    disk_new->setTransformationMatrix( transform );

    disk_new->setColor( color );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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
        getPrimitivePointer_private(UUID.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer_private(UUID.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer_private(UUID.at(i))->translate( center );

    }

    auto* disk_new = (new Disk(currentObjectID, UUID, Ndivs, texture_file, this));

    float T[16], transform[16];
    disk_new->getTransformationMatrix( transform );

    makeScaleMatrix(make_vec3(size.x,size.y,1.f),T);
    matmult(T,transform,transform);

    makeTranslationMatrix(center,T);
    matmult(T,transform,transform);

    disk_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    objects[currentObjectID] = disk_new;
    currentObjectID++;
    return currentObjectID-1;

}

uint Context::addPolymeshObject(const std::vector<uint> &UUIDs ){

    auto* polymesh_new = (new Polymesh(currentObjectID, UUIDs, "", this));

    polymesh_new->setColor( getPrimitivePointer_private( UUIDs.front())->getColor() );

    float T[16], transform[16];
    polymesh_new->getTransformationMatrix( transform );

    makeTranslationMatrix( getPrimitivePointer_private( UUIDs.front())->getVertices().front(),T);
    matmult(T,transform,transform);

    polymesh_new->setTransformationMatrix( transform );

    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->setParentObjectID(currentObjectID);
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

    auto* cone_new = (new Cone(currentObjectID, UUID, node0, node1, radius0, radius1, Ndivs, "", this));

    float T[16],  transform[16];
    cone_new->getTransformationMatrix( transform );

    makeTranslationMatrix(nodes.front(),T);
    matmult(T,transform,transform);

    cone_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
    }

    cone_new->setColor( color );

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

    auto* cone_new = (new Cone(currentObjectID, UUID, node0, node1, radius0, radius1, Ndivs, texturefile, this));

    float T[16],  transform[16];
    cone_new->getTransformationMatrix( transform );

    makeTranslationMatrix(nodes.front(),T);
    matmult(T,transform,transform);

    cone_new->setTransformationMatrix( transform );

    for( uint p : UUID){
        getPrimitivePointer_private(p)->setParentObjectID(currentObjectID);
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
                getPrimitivePointer_private( UUID[t] )->rotate( -rotation.elevation, "x" );
            }
            if( rotation.azimuth!=0.f ){
                getPrimitivePointer_private( UUID[t] )->rotate( -rotation.azimuth, "z" );
            }
            getPrimitivePointer_private( UUID[t] )->translate( center );

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

    addTexture( texturefile );
    std::vector<std::vector<bool> > alpha;
    int2 sz;
    if( textures.at(texturefile).hasTransparencyChannel() ){
        alpha = *textures.at(texturefile).getTransparencyData();
        sz = textures.at(texturefile).getSize();
    }

    for( uint j=0; j<subdiv.y; j++ ){
        for( uint i=0; i<subdiv.x; i++ ){

            subcenter = make_vec3(-0.5f*size.x+(float(i)+0.5f)*subsize.x,-0.5f*size.y+(float(j)+0.5f)*subsize.y,0);

            uv[0] = make_vec2(float(i)*uv_sub.x,float(j)*uv_sub.y);
            uv[1] = make_vec2(float(i+1)*uv_sub.x,float(j)*uv_sub.y);
            uv[2] = make_vec2(float(i+1)*uv_sub.x,float(j+1)*uv_sub.y);
            uv[3] = make_vec2(float(i)*uv_sub.x,float(j+1)*uv_sub.y);

            float solid_fraction;
            if( textures.at(texturefile).hasTransparencyChannel() ){
                int A = 0;
                int At = 0;

                int2 uv_min( floor(uv[0].x*float(sz.x-1)), floor(uv[0].y*float(sz.y-1)) );
                int2 uv_max( floor(uv[2].x*float(sz.x-1)), floor(uv[2].y*float(sz.y-1)) );

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

            auto* patch_new = (new Patch( texturefile, uv, solid_fraction, currentUUID ));

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

std::vector<uint> Context::addTube(uint Ndivs, const std::vector<vec3> &nodes, const std::vector<float> &radius, const std::vector<RGBcolor> &color ){

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
        getPrimitivePointer_private(UUIDs.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer_private(UUIDs.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer_private(UUIDs.at(i))->translate( center );

    }

    return UUIDs;

}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){

    std::vector<uint> UUIDs;
    UUIDs.resize(Ndivs);

    for( int i=0; i<Ndivs; i++ ){

        float dtheta = 2.f*float(M_PI)/float(Ndivs);

        UUIDs.at(i) = addTriangle( make_vec3(0,0,0), make_vec3(size.x*cosf(dtheta*float(i)),size.y*sinf(dtheta*float(i)),0), make_vec3(size.x*cosf(dtheta*float(i+1)),size.y*sinf(dtheta*float(i+1)),0), texture_file, make_vec2(0.5,0.5), make_vec2(0.5f*(1.f+cosf(dtheta*float(i))),0.5f*(1.f+sinf(dtheta*float(i)))), make_vec2(0.5f*(1.f+cosf(dtheta*float(i+1))),0.5f*(1.f+sinf(dtheta*float(i+1))))  );
        getPrimitivePointer_private(UUIDs.at(i))->rotate( rotation.elevation, "y" );
        getPrimitivePointer_private(UUIDs.at(i))->rotate( rotation.azimuth, "z" );
        getPrimitivePointer_private(UUIDs.at(i))->translate( center );

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

void Context::loadOData( pugi::xml_node p, uint ID ){

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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_INT,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_UINT,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_FLOAT,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_DOUBLE,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_VEC2,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_VEC3,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_VEC4,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_INT2,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_INT3,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_INT4,datav.size(),&datav[0]);
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
            setObjectData(ID,label,datav.front());
        }else if( datav.size()>1 ){
            setObjectData(ID,label,HELIOS_TYPE_STRING,datav.size(),&datav[0]);
        }

    }

}

void Context::loadOsubPData( pugi::xml_node p, uint ID ){

    std::vector<uint> prim_UUIDs = getObjectPointer(ID)->getPrimitiveUUIDs();

    int u;

    for (pugi::xml_node prim_data = p.child("primitive_data_int"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<int> datav;
            if (strlen(data_str) > 0) {
                std::istringstream data_stream(data_str);
                int tmp;
                while (data_stream >> tmp) {
                    datav.push_back(tmp);
                }
            }

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_INT, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_uint"); prim_data; prim_data = prim_data.next_sibling("primitive_data_uint")) {

        const char *label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<uint> datav;
            if (strlen(data_str) > 0) {
                std::istringstream data_stream(data_str);
                uint tmp;
                while (data_stream >> tmp) {
                    datav.push_back(tmp);
                }
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_UINT, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_float"); prim_data; prim_data = prim_data.next_sibling("primitive_data_float")){

        const char* label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<float> datav;
            if (strlen(data_str) > 0) {
                std::istringstream data_stream(data_str);
                float tmp;
                while (data_stream >> tmp) {
                    datav.push_back(tmp);
                }
            }

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_FLOAT, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_double"); prim_data; prim_data = prim_data.next_sibling("primitive_data_double")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<double> datav;
            if (strlen(data_str) > 0) {
                std::istringstream data_stream(data_str);
                double tmp;
                while (data_stream >> tmp) {
                    datav.push_back(tmp);
                }
            }

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_DOUBLE, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_vec2"); prim_data; prim_data = prim_data.next_sibling("primitive_data_vec2")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<vec2> datav;
            if (strlen(data_str) > 0) {
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

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_VEC2, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_vec3"); prim_data; prim_data = prim_data.next_sibling("primitive_data_vec3")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<vec3> datav;
            if (strlen(data_str) > 0) {
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

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_VEC3, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_vec4"); prim_data; prim_data = prim_data.next_sibling("primitive_data_vec4")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<vec4> datav;
            if (strlen(data_str) > 0) {
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

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_VEC4, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_int2"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int2")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<int2> datav;
            if (strlen(data_str) > 0) {
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

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_INT2, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_int3"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int3")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<int3> datav;
            if (strlen(data_str) > 0) {
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

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_INT3, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_int4"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int4")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<int4> datav;
            if (strlen(data_str) > 0) {
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

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_INT4, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_string"); prim_data; prim_data = prim_data.next_sibling("primitive_data_string")){

        const char* label = prim_data.attribute("label").value();

        u=0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {

            if( u>=prim_UUIDs.size() ){
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            const char *data_str = data.child_value();
            std::vector<std::string> datav;
            if (strlen(data_str) > 0) {
                std::istringstream data_stream(data_str);
                std::string tmp;
                while( data_stream >> tmp ){
                    datav.push_back(tmp);
                }
            }

            if( doesPrimitiveExist(prim_UUIDs.at(u)) ) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, HELIOS_TYPE_STRING, datav.size(), &datav[0]);
                }
            }
            u++;
        }
    }

}

std::vector<uint> Context::loadXML( const char* filename, bool quiet ){

    if( !quiet ) {
        std::cout << "Loading XML file: " << filename << "..." << std::flush;
    }

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
        if( !quiet ) {
            std::cout << "failed." << std::endl;
        }
        throw( std::runtime_error("ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags."));
    }

    //if primitives are added that belong to an object, store there UUIDs here so that we can make sure their UUIDs are consistent
    std::map<uint,std::vector<uint> > object_prim_UUIDs;

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

        // * Patch Object ID * //
        uint objID = 0;
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        if( !oid.empty() ){
            objID = std::stoi( oid );
        }

        // * Patch Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char* transform_str = transform_node.child_value();
        if( strlen(transform_str)==0 ){
            makeIdentityMatrix(transform);
        }else{
            std::istringstream stream(transform_str);
            float tmp;
            int i=0;
            while( stream >> tmp ){
                transform[i] = tmp;
                i++;
            }
            if( i!=16 ){
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
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
        getPrimitivePointer_private(ID)->setTransformationMatrix(transform);

        if( objID>0 ) {
            object_prim_UUIDs[objID].push_back(ID);
        }

        UUID.push_back(ID);

        // * Primitive Data * //

        loadPData( p, ID );

    }//end patches

    //-------------- TRIANGLES ---------------//

    //looping over any triangles specified in XML file
    for (pugi::xml_node tri = helios.child("triangle"); tri; tri = tri.next_sibling("triangle")){

        // * Triangle Object ID * //
        uint objID = 0;
        pugi::xml_node objID_node = tri.child("objID");
        std::string oid = deblank(objID_node.child_value());
        if( !oid.empty() ){
            objID = std::stoi( oid );
        }

        // * Triangle Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = tri.child("transform");

        const char* transform_str = transform_node.child_value();
        if( strlen(transform_str)==0 ){
            makeIdentityMatrix(transform);
        }else{
            std::istringstream stream(transform_str);
            float tmp;
            int i=0;
            while( stream >> tmp ){
                transform[i] = tmp;
                i++;
            }
            if( i!=16 ){
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
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

        const char* color_str = color_node.child_value();
        if( strlen(color_str)==0 ){
            color = make_RGBAcolor(0,0,0,1);//assume default color of black
        }else{
            color=string2RGBcolor(color_str);
        }

        std::vector<vec3> vert_pos;
        vert_pos.resize(3);
        vert_pos.at(0) = make_vec3( 0.f, 0.f, 0.f);
        vert_pos.at(1) = make_vec3( 0.f, 1.f, 0.f);
        vert_pos.at(2) = make_vec3( 1.f, 1.f, 0.f);

        // * Add the Triangle * //
        if( strcmp(texture_file.c_str(),"none")==0 || uv.empty() ){
            ID = addTriangle( vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), color );
        }else{
            ID = addTriangle( vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), texture_file.c_str(), uv.at(0), uv.at(1), uv.at(2) );
        }
        getPrimitivePointer_private(ID)->setTransformationMatrix(transform);

        if( objID>0 ) {
            object_prim_UUIDs[objID].push_back(ID);
        }

        UUID.push_back(ID);

        // * Primitive Data * //

        loadPData( tri, ID );

    }

    //-------------- VOXELS ---------------//
    for (pugi::xml_node p = helios.child("voxel"); p; p = p.next_sibling("voxel")){

        // * Voxel Object ID * //
        uint objID = 0;
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        if( !oid.empty() ){
            objID = std::stoi( oid );
        }

        // * Voxel Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char* transform_str = transform_node.child_value();
        if( strlen(transform_str)==0 ){
            makeIdentityMatrix(transform);
        }else{
            std::istringstream stream(transform_str);
            float tmp;
            int i=0;
            while( stream >> tmp ){
                transform[i] = tmp;
                i++;
            }
            if( i!=16 ){
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Voxel Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char* color_str = color_node.child_value();
        if( strlen(color_str)==0 ){
            color = make_RGBAcolor(0,0,0,1);//assume default color of black
        }else{
            color=string2RGBcolor(color_str);
        }

        // * Add the Voxel * //
        ID = addVoxel( make_vec3(0,0,0), make_vec3(0,0,0), 0, color );
        getPrimitivePointer_private(ID)->setTransformationMatrix(transform);

        if( objID>0 ) {
            object_prim_UUIDs[objID].push_back(ID);
        }

        UUID.push_back(ID);

        // * Primitive Data * //

        loadPData( p, ID );

    }

    //-------------- COMPOUND OBJECTS ---------------//

    //-------------- TILES ---------------//
    for (pugi::xml_node p = helios.child("tile"); p; p = p.next_sibling("tile")) {

        // * Tile Object ID * //
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        uint objID = std::stoi( oid );

        // * Tile Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char *transform_str = transform_node.child_value();
        if (strlen(transform_str) == 0) {
            makeIdentityMatrix(transform);
        } else {
            std::istringstream stream(transform_str);
            float tmp;
            int i = 0;
            while (stream >> tmp) {
                transform[i] = tmp;
                i++;
            }
            if (i != 16) {
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Tile Texture * //
        std::string texture_file;
        pugi::xml_node texture_node = p.child("texture");
        std::string texfile = deblank(texture_node.child_value());
        if (texfile.empty()) {
            texture_file = "none";
        } else {
            texture_file = texfile;
        }

        // * Tile Texture (u,v) Coordinates * //
        std::vector<vec2> uv;
        pugi::xml_node uv_node = p.child("textureUV");
        const char *texUV = uv_node.child_value();
        if (strlen(texUV) > 0) {
            std::istringstream uv_stream(texUV);
            std::vector<float> tmp;
            tmp.resize(2);
            int c = 0;
            while (uv_stream >> tmp.at(c)) {
                c++;
                if (c == 2) {
                    uv.push_back(make_vec2(tmp.at(0), tmp.at(1)));
                    c = 0;
                }
            }
            if (c != 0) {
                std::cerr << "WARNING (loadXML): textureUV for tile does not contain an even number of elements. Skipping..." << std::endl;
                uv.resize(0);
            }
            if (uv.size() != 4) {
                std::cerr << "WARNING (loadXML): textureUV for tile does not contain four pairs of (u,v) coordinates. Skipping..." << std::endl;
                uv.resize(0);
            }
        }

        // * Tile Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if ( strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Tile Subdivisions * //
        int2 subdiv;
        pugi::xml_node subdiv_node = p.child("subdivisions");
        const char* subdiv_str = subdiv_node.child_value();
        if (strlen(subdiv_str) == 0) {
            std::cerr << "WARNING (loadXML): Number of subdivisions for tile was not provided. Assuming 1x1." << std::endl;
            subdiv = make_int2(1,1);
        } else {
            subdiv = string2int2(subdiv_str);
        }

        //Create a dummy patch in order to get the center, size, and rotation based on transformation matrix
        Patch patch( make_RGBAcolor(0,0,0,0), 0 );
        patch.setTransformationMatrix(transform);

        // * Add the Tile * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if( strlen(color_str) == 0 ){
                ID = addTileObject(patch.getCenter(), patch.getSize(), cart2sphere(patch.getNormal()), subdiv );
            }else {
                ID = addTileObject(patch.getCenter(), patch.getSize(), cart2sphere(patch.getNormal()), subdiv, make_RGBcolor(color.r, color.g, color.b));
            }
        } else {
            ID = addTileObject(patch.getCenter(), patch.getSize(), cart2sphere(patch.getNormal()), subdiv, texture_file.c_str());
        }

        deletePrimitive(getObjectPrimitiveUUIDs(ID)); // \todo This is fairly inefficient, it would be nice to have a way to do this without having to create and delete a bunch of primitives.
        assert(object_prim_UUIDs.find(objID) != object_prim_UUIDs.end());
        getObjectPointer(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));

        // * Tile Sub-Patch Data * //

        loadOsubPData(p,ID);

        // * Tile Object Data * //

        loadOData(p,ID);

//        UUID.push_back(ID);

    }//end tiles

    //-------------- SPHERES ---------------//
    for (pugi::xml_node p = helios.child("sphere"); p; p = p.next_sibling("sphere")) {

        // * Sphere Object ID * //
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        uint objID = std::stoi( oid );

        // * Sphere Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char *transform_str = transform_node.child_value();
        if (strlen(transform_str) == 0) {
            makeIdentityMatrix(transform);
        } else {
            std::istringstream stream(transform_str);
            float tmp;
            int i = 0;
            while (stream >> tmp) {
                transform[i] = tmp;
                i++;
            }
            if (i != 16) {
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Sphere Texture * //
        std::string texture_file;
        pugi::xml_node texture_node = p.child("texture");
        std::string texfile = deblank(texture_node.child_value());
        if (texfile.empty()) {
            texture_file = "none";
        } else {
            texture_file = texfile;
        }

        // * Sphere Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if ( strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Sphere Subdivisions * //
        uint subdiv;
        pugi::xml_node subdiv_node = p.child("subdivisions");
        const char* subdiv_str = subdiv_node.child_value();
        if (strlen(subdiv_str) == 0) {
            std::cerr << "WARNING (loadXML): Number of subdivisions for sphere was not provided. Assuming 5." << std::endl;
            subdiv = 5;
        } else {
            subdiv = std::stoi(subdiv_str);
        }

        //Create a dummy sphere in order to get the center and radius based on transformation matrix
        std::vector<uint> empty;
        Sphere sphere( 0, empty, 3, "", this );
        sphere.setTransformationMatrix(transform);

        // * Add the Sphere * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if( strlen(color_str) == 0 ){
                ID = addSphereObject( subdiv, sphere.getCenter(), sphere.getRadius() );
            }else {
                ID = addSphereObject( subdiv, sphere.getCenter(), sphere.getRadius(), make_RGBcolor(color.r, color.g, color.b) );
            }
        } else {
            ID = addSphereObject( subdiv, sphere.getCenter(), sphere.getRadius(), texture_file.c_str());
        }

        deletePrimitive(getObjectPrimitiveUUIDs(ID));
        assert(object_prim_UUIDs.find(objID) != object_prim_UUIDs.end());
        getObjectPointer(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));

        // * Sphere Sub-Triangle Data * //

        loadOsubPData(p,ID);

        // * Sphere Object Data * //

        loadOData(p,ID);

    }//end spheres

    //-------------- TUBES ---------------//
    for (pugi::xml_node p = helios.child("tube"); p; p = p.next_sibling("tube")) {

        // * Tube Object ID * //
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        uint objID = std::stoi( oid );

        // * Tube Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char *transform_str = transform_node.child_value();
        if (strlen(transform_str) == 0) {
            makeIdentityMatrix(transform);
        } else {
            std::istringstream stream(transform_str);
            float tmp;
            int i = 0;
            while (stream >> tmp) {
                transform[i] = tmp;
                i++;
            }
            if (i != 16) {
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Tube Texture * //
        std::string texture_file;
        pugi::xml_node texture_node = p.child("texture");
        std::string texfile = deblank(texture_node.child_value());
        if (texfile.empty()) {
            texture_file = "none";
        } else {
            texture_file = texfile;
        }

        // * Tube Subdivisions * //
        uint subdiv;
        pugi::xml_node subdiv_node = p.child("subdivisions");
        const char* subdiv_str = subdiv_node.child_value();
        if (strlen(subdiv_str) == 0) {
            std::cerr << "WARNING (loadXML): Number of subdivisions for tube was not provided. Assuming 5." << std::endl;
            subdiv = 5;
        } else {
            subdiv = std::stoi(subdiv_str);
        }

        // * Tube Nodes * //

        pugi::xml_node nodes_node = p.child("nodes");
        const char* nodes_str = nodes_node.child_value();

        std::vector<vec3> nodes;
        if (strlen(nodes_str) > 0) {
            std::istringstream data_stream(nodes_str);
            std::vector<float> tmp;
            tmp.resize(3);
            int c = 0;
            while( data_stream >> tmp.at(c) ){
                c++;
                if( c==3 ){
                    nodes.push_back(make_vec3(tmp.at(0),tmp.at(1),tmp.at(2)));
                    c=0;
                }
            }
        }

        // * Tube Radius * //

        pugi::xml_node radii_node = p.child("radius");
        const char* radii_str = radii_node.child_value();

        std::vector<float> radii;
        if (strlen(radii_str) > 0) {
            std::istringstream data_stream(radii_str);
            float tmp;
            int c = 0;
            while( data_stream >> tmp ){
                radii.push_back(tmp);
            }
        }

        // * Tube Color * //

        pugi::xml_node color_node = p.child("color");
        const char* color_str = color_node.child_value();

        std::vector<RGBcolor> colors;
        if (strlen(color_str) > 0) {
            std::istringstream data_stream(color_str);
            std::vector<float> tmp;
            tmp.resize(3);
            int c = 0;
            while( data_stream >> tmp.at(c) ){
                c++;
                if( c==3 ){
                    colors.push_back(make_RGBcolor(tmp.at(0),tmp.at(1),tmp.at(2)));
                    c=0;
                }
            }
        }

        // * Add the Tube * //
        if( texture_file == "none" ) {
            ID = addTubeObject(subdiv, nodes, radii, colors);
        }else{
            ID = addTubeObject(subdiv, nodes, radii, texture_file.c_str());
        }

        getObjectPointer(ID)->setTransformationMatrix(transform);

        deletePrimitive(getObjectPrimitiveUUIDs(ID));
        assert(object_prim_UUIDs.find(objID) != object_prim_UUIDs.end());
        getObjectPointer(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));

        // * Tube Sub-Triangle Data * //

        loadOsubPData(p,ID);

        // * tube Object Data * //

        loadOData(p,ID);

    }//end tubes

    //-------------- BOXES ---------------//
    for (pugi::xml_node p = helios.child("box"); p; p = p.next_sibling("box")) {

        // * Box Object ID * //
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        uint objID = std::stoi( oid );

        // * Box Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char *transform_str = transform_node.child_value();
        if (strlen(transform_str) == 0) {
            makeIdentityMatrix(transform);
        } else {
            std::istringstream stream(transform_str);
            float tmp;
            int i = 0;
            while (stream >> tmp) {
                transform[i] = tmp;
                i++;
            }
            if (i != 16) {
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Box Texture * //
        std::string texture_file;
        pugi::xml_node texture_node = p.child("texture");
        std::string texfile = deblank(texture_node.child_value());
        if (texfile.empty()) {
            texture_file = "none";
        } else {
            texture_file = texfile;
        }

        // * Box Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if ( strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Box Subdivisions * //
        int3 subdiv;
        pugi::xml_node subdiv_node = p.child("subdivisions");
        const char* subdiv_str = subdiv_node.child_value();
        if (strlen(subdiv_str) == 0) {
            std::cerr << "WARNING (loadXML): Number of subdivisions for box was not provided. Assuming 1." << std::endl;
            subdiv = make_int3(1,1,1);
        } else {
            subdiv = string2int3(subdiv_str);
        }

        //Create a dummy box in order to get the center and size based on transformation matrix
        std::vector<uint> empty;
        Box box( 0, empty, make_int3(1,1,1), "", this );
        box.setTransformationMatrix(transform);

        // * Add the box * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if( strlen(color_str) == 0 ){
                ID = addBoxObject( box.getCenter(), box.getSize(), subdiv  );
            }else {
                ID = addBoxObject( box.getCenter(), box.getSize(), subdiv, make_RGBcolor(color.r, color.g, color.b) );
            }
        } else {
            ID = addBoxObject( box.getCenter(), box.getSize(), subdiv, texture_file.c_str());
        }

        deletePrimitive(getObjectPrimitiveUUIDs(ID));
        assert(object_prim_UUIDs.find(objID)!=object_prim_UUIDs.end());
        getObjectPointer(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));

        // * Box Sub-Patch Data * //

        loadOsubPData(p,ID);

        // * Box Object Data * //

        loadOData(p,ID);

    }//end boxes

    //-------------- DISKS ---------------//
    for (pugi::xml_node p = helios.child("disk"); p; p = p.next_sibling("disk")) {

        // * Disk Object ID * //
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        uint objID = std::stoi( oid );

        // * Disk Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char *transform_str = transform_node.child_value();
        if (strlen(transform_str) == 0) {
            makeIdentityMatrix(transform);
        } else {
            std::istringstream stream(transform_str);
            float tmp;
            int i = 0;
            while (stream >> tmp) {
                transform[i] = tmp;
                i++;
            }
            if (i != 16) {
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Disk Texture * //
        std::string texture_file;
        pugi::xml_node texture_node = p.child("texture");
        std::string texfile = deblank(texture_node.child_value());
        if (texfile.empty()) {
            texture_file = "none";
        } else {
            texture_file = texfile;
        }

        // * Disk Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if ( strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Disk Subdivisions * //
        uint subdiv;
        pugi::xml_node subdiv_node = p.child("subdivisions");
        const char* subdiv_str = subdiv_node.child_value();
        if (strlen(subdiv_str) == 0) {
            std::cerr << "WARNING (loadXML): Number of subdivisions for disk was not provided. Assuming 5." << std::endl;
            subdiv = 5;
        } else {
            subdiv = std::stoi(subdiv_str);
        }

        //Create a dummy disk in order to get the center and size based on transformation matrix
        std::vector<uint> empty;
        Disk disk( 0, empty, 1, "", this );
        disk.setTransformationMatrix(transform);

        // * Add the disk * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if( strlen(color_str) == 0 ){
                ID = addDiskObject( subdiv, disk.getCenter(), disk.getSize() );
            }else {
                ID = addDiskObject( subdiv, disk.getCenter(), disk.getSize(), nullrotation, make_RGBcolor(color.r, color.g, color.b) );
            }
        } else {
            ID = addDiskObject( subdiv, disk.getCenter(), disk.getSize(), nullrotation, texture_file.c_str());
        }

        deletePrimitive(getObjectPrimitiveUUIDs(ID));
        assert(object_prim_UUIDs.find(objID)!=object_prim_UUIDs.end());
        getObjectPointer(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));

        // * Disk Sub-Triangle Data * //

        loadOsubPData(p,ID);

        // * Disk Object Data * //

        loadOData(p,ID);

    }//end disks

    //-------------- CONES ---------------//
    for (pugi::xml_node p = helios.child("cone"); p; p = p.next_sibling("cone")) {

        // * Cone Object ID * //
        pugi::xml_node objID_node = p.child("objID");
        std::string oid = deblank(objID_node.child_value());
        uint objID = std::stoi( oid );

        // * Cone Transformation Matrix * //
        float transform[16];
        pugi::xml_node transform_node = p.child("transform");

        const char *transform_str = transform_node.child_value();
        if (strlen(transform_str) == 0) {
            makeIdentityMatrix(transform);
        } else {
            std::istringstream stream(transform_str);
            float tmp;
            int i = 0;
            while (stream >> tmp) {
                transform[i] = tmp;
                i++;
            }
            if (i != 16) {
                if( !quiet ) {
                    std::cout << "WARNING (Context::loadXML): Transformation matrix does not have 16 elements. Assuming identity matrix." << std::endl;
                }
                makeIdentityMatrix(transform);
            }
        }

        // * Cone Texture * //
        std::string texture_file;
        pugi::xml_node texture_node = p.child("texture");
        std::string texfile = deblank(texture_node.child_value());
        if (texfile.empty()) {
            texture_file = "none";
        } else {
            texture_file = texfile;
        }

        // * Cone Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if ( strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Cone Subdivisions * //
        uint subdiv;
        pugi::xml_node subdiv_node = p.child("subdivisions");
        const char* subdiv_str = subdiv_node.child_value();
        if (strlen(subdiv_str) == 0) {
            std::cerr << "WARNING (loadXML): Number of subdivisions for cone was not provided. Assuming 5." << std::endl;
            subdiv = 5;
        } else {
            subdiv = std::stoi(subdiv_str);
        }

        // * Cone Nodes * //

        pugi::xml_node nodes_node = p.child("nodes");
        const char* nodes_str = nodes_node.child_value();

        std::vector<vec3> nodes;
        if (strlen(nodes_str) > 0) {
            std::istringstream data_stream(nodes_str);
            std::vector<float> tmp;
            tmp.resize(3);
            int c = 0;
            while( data_stream >> tmp.at(c) ){
                c++;
                if( c==3 ){
                    nodes.push_back(make_vec3(tmp.at(0),tmp.at(1),tmp.at(2)));
                    break;
                }
            }
            if( c!=3 ){
                throw(std::runtime_error("ERROR (loadXML): Loading of cone failed. Cone end nodes must be specified as pairs of 3 x,y,z coordinates."));
            }
            c = 0;
            while( data_stream >> tmp.at(c) ){
                c++;
                if( c==3 ){
                    nodes.push_back(make_vec3(tmp.at(0),tmp.at(1),tmp.at(2)));
                    break;
                }
            }
            if( c!=3 ){
                throw(std::runtime_error("ERROR (loadXML): Loading of cone failed. Cone end nodes must be specified as pairs of 3 x,y,z coordinates."));
            }
        }

        // * Cone Radius * //

        pugi::xml_node radii_node = p.child("radius");
        const char* radii_str = radii_node.child_value();

        std::vector<float> radii(2);
        if (strlen(radii_str) > 0) {
            std::istringstream data_stream(radii_str);
            data_stream >> radii.at(0);
            data_stream >> radii.at(1);
        }

        // * Add the Cone * //
        if( texture_file == "none" ){
            ID = addConeObject( subdiv, nodes.at(0), nodes.at(1), radii.at(0), radii.at(1), make_RGBcolor(color.r,color.g,color.b));
        }else {
            ID = addConeObject(subdiv, nodes.at(0), nodes.at(1), radii.at(0), radii.at(1), texture_file.c_str());
        }

        getObjectPointer(ID)->setTransformationMatrix(transform);

        deletePrimitive(getObjectPrimitiveUUIDs(ID));
        assert(object_prim_UUIDs.find(objID)!=object_prim_UUIDs.end());
        getObjectPointer(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));

        // * Tube Sub-Triangle Data * //

        loadOsubPData(p,ID);

        // * tube Object Data * //

        loadOData(p,ID);

    }//end cones

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

    if( !quiet ) {
        std::cout << "done." << std::endl;
    }

    return UUID;

}

std::vector<std::string> Context::getLoadedXMLFiles() {
    return XMLfiles;
}

void Context::writeDataToXMLstream( const char* data_group, const std::vector<std::string> &data_labels, void* ptr, std::ofstream &outfile ) const{

    for(const auto& label : data_labels ) {

        size_t dsize;
        HeliosDataType dtype;

        if (strcmp(data_group, "primitive") == 0) {
            dsize = ((Primitive *) ptr)->getPrimitiveDataSize(label.c_str());
            dtype = ((Primitive *) ptr)->getPrimitiveDataType(label.c_str());
        } else if (strcmp(data_group, "object") == 0) {
            dsize = ((CompoundObject *) ptr)->getObjectDataSize(label.c_str());
            dtype = ((CompoundObject *) ptr)->getObjectDataType(label.c_str());
        } else if (strcmp(data_group, "global") == 0) {
            dsize = getGlobalDataSize(label.c_str());
            dtype = getGlobalDataType(label.c_str());
        } else {
            throw( std::runtime_error( "ERROR (writeDataToXMLstream): unknown data group argument of " + std::string(data_group) + ". Must be one of primitive, object, or global."));
        }

        if (dtype == HELIOS_TYPE_UINT) {
            outfile << "\t<data_uint label=\"" << label << "\">" << std::flush;
            std::vector<uint> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_uint>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT) {
            outfile << "\t<data_int label=\"" << label << "\">" << std::flush;
            std::vector<int> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int>" << std::endl;
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            outfile << "\t<data_float label=\"" << label << "\">" << std::flush;
            std::vector<float> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_float>" << std::endl;
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            outfile << "\t<data_double label=\"" << label << "\">" << std::flush;
            std::vector<double> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_double>" << std::endl;
        } else if (dtype == HELIOS_TYPE_VEC2) {
            outfile << "\t<data_vec2 label=\"" << label << "\">" << std::flush;
            std::vector<vec2> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_vec2>" << std::endl;
        } else if (dtype == HELIOS_TYPE_VEC3) {
            outfile << "\t<data_vec3 label=\"" << label << "\">" << std::flush;
            std::vector<vec3> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_vec3>" << std::endl;
        } else if (dtype == HELIOS_TYPE_VEC4) {
            outfile << "\t<data_vec4 label=\"" << label << "\">" << std::flush;
            std::vector<vec4> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << " " << data.at(j).w
                        << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_vec4>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT2) {
            outfile << "\t<data_int2 label=\"" << label << "\">" << std::flush;
            std::vector<int2> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int2>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT3) {
            outfile << "\t<data_int3 label=\"" << label << "\">" << std::flush;
            std::vector<int3> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int3>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT4) {
            outfile << "\t<data_int3 label=\"" << label << "\">" << std::flush;
            std::vector<int4> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << " " << data.at(j).w
                        << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int4>" << std::endl;
        } else if (dtype == HELIOS_TYPE_STRING) {
            outfile << "\t<data_string label=\"" << label << "\">" << std::flush;
            std::vector<std::string> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_string>" << std::endl;
        }

    }

}

void Context::writeXML( const char* filename, bool quiet ) const{

    if( !quiet ) {
        std::cout << "Writing XML file " << filename << "..." << std::flush;
    }

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

    // -- primitives -- //

    for(auto primitive : primitives){

        uint p = primitive.first;

        Primitive* prim = getPrimitivePointer_private(p);

        uint parent_objID = prim->getParentObjectID();

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

        if( parent_objID>0 ){
            outfile << "\t<objID>" << parent_objID << "</objID>" << std::endl;
        }

        outfile << "\t<color>" << color.r << " " << color.g << " " << color.b << " " << color.a << "</color>" << std::endl;
        if( prim->hasTexture() ){
            outfile << "\t<texture>" << texture_file << "</texture>" << std::endl;
        }

        if( !pdata.empty() ){
            writeDataToXMLstream( "primitive", pdata, prim, outfile );
        }

        //Patches
        if( prim->getType()==PRIMITIVE_TYPE_PATCH ){

            Patch* patch = getPatchPointer_private(p);
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

            std::vector<vec2> uv = getTrianglePointer_private(p)->getTextureUV();
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

    // -- objects -- //

    for( const auto &object : objects ){

        uint o = object.first;

        CompoundObject* obj = object.second;

        RGBAcolor color = obj->getColorRGBA();

        std::string texture_file = obj->getTextureFile();

        std::vector<std::string> odata = obj->listObjectData();

//        std::vector<uint> prim_UUIDs = obj->getPrimitiveUUIDs();

        if( obj->getObjectType()==OBJECT_TYPE_TILE ){
            outfile << "   <tile>" << std::endl;
        }else if( obj->getObjectType()==OBJECT_TYPE_BOX ){
            outfile << "   <box>" << std::endl;
        }else if( obj->getObjectType()==OBJECT_TYPE_CONE ){
            outfile << "   <cone>" << std::endl;
        }else if( obj->getObjectType()==OBJECT_TYPE_DISK ){
            outfile << "   <disk>" << std::endl;
        }else if( obj->getObjectType()==OBJECT_TYPE_SPHERE ){
            outfile << "   <sphere>" << std::endl;
        }else if( obj->getObjectType()==OBJECT_TYPE_TUBE ){
            outfile << "   <tube>" << std::endl;
        }else if( obj->getObjectType()==OBJECT_TYPE_POLYMESH ){
            outfile << "   <polymesh>" << std::endl;
        }

        outfile << "\t<objID>" << o << "</objID>" << std::endl;
        if( !(obj->getObjectType()==OBJECT_TYPE_TUBE) ){
            outfile << "\t<color>" << color.r << " " << color.g << " " << color.b << " " << color.a << "</color>" << std::endl;
        }
        if( obj->hasTexture() ){
            outfile << "\t<texture>" << texture_file << "</texture>" << std::endl;
        }

//        outfile << "\t<UUIDs>" << std::endl;
//        for( uint UUID : prim_UUIDs ){
//            outfile << "\t\t" << UUID << std::endl;
//        }
//        outfile << "\t</UUIDs>" << std::endl;

        if( !odata.empty() ){
            writeDataToXMLstream( "object", odata, obj, outfile );
        }

        std::vector<std::string> pdata_labels;
        std::vector<HeliosDataType> pdata_types;
        std::vector<uint> primitiveUUIDs = obj->getPrimitiveUUIDs();
        for( uint UUID : primitiveUUIDs ){
            std::vector<std::string> labels = getPrimitivePointer_private(UUID)->listPrimitiveData();
            for( int i=0; i<labels.size(); i++ ){
                if( find(pdata_labels.begin(),pdata_labels.end(),labels.at(i)) == pdata_labels.end() ){
                    pdata_labels.push_back(labels.at(i));
                    pdata_types.push_back(getPrimitiveDataType(UUID,labels.at(i).c_str()));
                }
            }
        }
        for( size_t l=0; l<pdata_labels.size(); l++ ) {
            if( pdata_types.at(l)==HELIOS_TYPE_FLOAT ) {
                outfile << "\t<primitive_data_float " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<float> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i) << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_float>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_DOUBLE ) {
                outfile << "\t<primitive_data_double " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<double> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i) << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_double>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_UINT ) {
                outfile << "\t<primitive_data_uint " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<uint> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i) << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_uint>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_INT ) {
                outfile << "\t<primitive_data_int " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i) << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_INT2 ) {
                outfile << "\t<primitive_data_int2 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int2> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i).x << " " << data.at(i).y << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int2>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_INT3 ) {
                outfile << "\t<primitive_data_int3 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int3> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i).x << " " << data.at(i).y << " " << data.at(i).z << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int3>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_INT4 ) {
                outfile << "\t<primitive_data_int4 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int4> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i).x << " " << data.at(i).y << " " << data.at(i).z << " " << data.at(i).w << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int4>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_VEC2 ) {
                outfile << "\t<primitive_data_vec2 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<vec2> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i).x << " " << data.at(i).y << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_vec2>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_VEC3 ) {
                outfile << "\t<primitive_data_vec3 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<vec3> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i).x << " " << data.at(i).y << " " << data.at(i).z << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_vec3>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_VEC4 ) {
                outfile << "\t<primitive_data_vec4 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<vec4> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i).x << " " << data.at(i).y << " " << data.at(i).z << " " << data.at(i).w << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_vec4>" << std::endl;
            }else if( pdata_types.at(l)==HELIOS_TYPE_STRING ) {
                outfile << "\t<primitive_data_string " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<std::string> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (size_t i = 0; i < data.size(); i++) {
                            outfile << data.at(i) << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_string>" << std::endl;
            }


        }

        //Tiles
        if( obj->getObjectType()==OBJECT_TYPE_TILE ){

            Tile* tile = getTileObjectPointer(o);

            float transform[16];
            tile->getTransformationMatrix(transform);

            int2 subdiv = tile->getSubdivisionCount();
            outfile << "\t<subdivisions>" << subdiv.x << " " << subdiv.y << "</subdivisions>" << std::endl;

            outfile << "\t<transform> ";
            for(float i : transform){
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            outfile << "   </tile>" << std::endl;

        //Spheres
        }else if( obj->getObjectType()==OBJECT_TYPE_SPHERE ){

            Sphere* sphere = getSphereObjectPointer(o);

            float transform[16];
            sphere->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for(float i : transform){
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = sphere->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            outfile << "   </sphere>" << std::endl;

        //Tubes
        }else if( obj->getObjectType()==OBJECT_TYPE_TUBE ) {

            Tube *tube = getTubeObjectPointer(o);

            float transform[16];
            tube->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = tube->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            std::vector<vec3> nodes = tube->getNodes();
            std::vector<float> radius = tube->getNodeRadii();

            assert(nodes.size() == radius.size());
            outfile << "\t<nodes> " << std::endl;
            for (int i = 0; i < nodes.size(); i++) {
                outfile << "\t\t" << nodes.at(i).x << " " << nodes.at(i).y << " " << nodes.at(i).z << std::endl;
            }
            outfile << "\t</nodes> " << std::endl;
            outfile << "\t<radius> " << std::endl;
            for (int i = 0; i < radius.size(); i++) {
                outfile << "\t\t" << radius.at(i) << std::endl;
            }
            outfile << "\t</radius> " << std::endl;

            if( texture_file.empty() ) {
                std::vector<RGBcolor> colors = tube->getNodeColors();

                outfile << "\t<color> " << std::endl;
                for (int i = 0; i < colors.size(); i++) {
                    outfile << "\t\t" << colors.at(i).r << " " << colors.at(i).g << " " << colors.at(i).b << std::endl;
                }
                outfile << "\t</color> " << std::endl;

            }

            outfile << "   </tube>" << std::endl;

        //Boxes
        }else if( obj->getObjectType()==OBJECT_TYPE_BOX ) {

            Box *box = getBoxObjectPointer(o);

            float transform[16];
            box->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            int3 subdiv = box->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv.x << " " << subdiv.y << " " << subdiv.z << " </subdivisions>" << std::endl;

            outfile << "   </box>" << std::endl;

        //Disks
        }else if( obj->getObjectType()==OBJECT_TYPE_DISK ) {

            Disk *disk = getDiskObjectPointer(o);

            float transform[16];
            disk->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = disk->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            outfile << "   </disk>" << std::endl;

        //Cones
        }else if( obj->getObjectType()==OBJECT_TYPE_CONE ) {

            Cone *cone = getConeObjectPointer(o);

            float transform[16];
            cone->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = cone->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            std::vector<vec3> nodes = cone->getNodes();
            std::vector<float> radius = cone->getNodeRadii();

            assert(nodes.size() == radius.size());
            outfile << "\t<nodes> " << std::endl;
            for (int i = 0; i < nodes.size(); i++) {
                outfile << "\t\t" << nodes.at(i).x << " " << nodes.at(i).y << " " << nodes.at(i).z << std::endl;
            }
            outfile << "\t</nodes> " << std::endl;
            outfile << "\t<radius> " << std::endl;
            for (int i = 0; i < radius.size(); i++) {
                outfile << "\t\t" << radius.at(i) << std::endl;
            }
            outfile << "\t</radius> " << std::endl;

            outfile << "   </cone>" << std::endl;

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

    if( !quiet ) {
        std::cout << "done." << std::endl;
    }
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

        std::vector<vec3> vertices = getPrimitivePointer_private(p)->getVertices();
        PrimitiveType type = getPrimitivePointer_private(p)->getType();
        RGBcolor C = getPrimitivePointer_private(p)->getColor();
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
            vertices.emplace_back(verts);

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
            texture_uv.emplace_back(uv);

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

                        vec3 normal = getPrimitivePointer_private(ID)->getNormal();

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

        std::vector < vec3 > vertices = getPrimitivePointer_private(p)->getVertices();
        PrimitiveType type = getPrimitivePointer_private(p)->getType();
        RGBcolor C = getPrimitivePointer_private(p)->getColor();

        if (type == PRIMITIVE_TYPE_TRIANGLE) {

            faces.push_back(make_int3( (int)vertex_count, (int)vertex_count + 1, (int)vertex_count + 2));
            colors.push_back(C);
            for (int i = 0; i < 3; i++) {
                verts.push_back(vertices.at(i));
                vertex_count++;
            }

            std::vector < vec2 > uv_v = getTrianglePointer_private(p)->getTextureUV();
            if (getTrianglePointer_private(p)->hasTexture()) {
                uv_inds.push_back(make_int3( (int)uv_count, (int)uv_count + 1, (int)uv_count + 2));
                texture_list.push_back(getTrianglePointer_private(p)->getTextureFile());
                for (int i = 0; i < 3; i++) {
                    uv.push_back( uv_v.at(i) );
                    uv_count++;
                }
            } else {
                texture_list.emplace_back("");
                uv_inds.push_back(make_int3(-1, -1, -1));
            }

        } else if (type == PRIMITIVE_TYPE_PATCH) {
            faces.push_back(make_int3( (int)vertex_count, (int)vertex_count + 1, (int)vertex_count + 2));
            faces.push_back(make_int3( (int)vertex_count, (int)vertex_count + 2, (int)vertex_count + 3));
            colors.push_back(C);
            colors.push_back(C);
            for (int i = 0; i < 4; i++) {
                verts.push_back(vertices.at(i));
                vertex_count++;
            }
            std::vector < vec2 > uv_v;
            std::string texturefile;
            uv_v = getPatchPointer_private(p)->getTextureUV();
            texturefile = getPatchPointer_private(p)->getTextureFile();

            if (getPatchPointer_private(p)->hasTexture()) {
                texture_list.push_back(texturefile);
                texture_list.push_back(texturefile);
                uv_inds.push_back(make_int3( (int)uv_count, (int)uv_count + 1, (int)uv_count + 2));
                uv_inds.push_back(make_int3( (int)uv_count, (int)uv_count + 2, (int)uv_count + 3));
                if (uv_v.empty()) {  //default (u,v)
                    uv.push_back( make_vec2(0, 1) );
                    uv.push_back( make_vec2(1, 1) );
                    uv.push_back( make_vec2(1, 0) );
                    uv.push_back( make_vec2(0, 0) );
                    uv_count += 4;
                } else {  //custom (u,v)
                    for (int i = 0; i < 4; i++) {
                        uv.push_back( uv_v.at(i) );
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
        file << "v " << vert.x << " " << vert.y << " " << vert.z << std::endl;
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
            if ( !(texture_list.at(f) != texture_list.at(exsit_mtl_list2.at(index)))) {
                mtl_exist_flag = true;
                mtl_index = index;
                mtl_index_f = exsit_mtl_list2[index];
                break;
            }
        }
        if ( !mtl_exist_flag) {

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

void Context::writePrimitiveData( std::string filename, const std::vector<std::string> &column_format, bool print_header ) const{
    writePrimitiveData(filename,column_format,getAllUUIDs(),print_header);
}

void Context::writePrimitiveData( std::string filename, const std::vector<std::string> &column_format, const std::vector<uint> &UUIDs, bool print_header ) const{

    std::ofstream file(filename);

    if( print_header ){
        for( const auto &label : column_format ) {
            file << label << " ";
        }
        file.seekp(-1, std::ios_base::end);
        file << "\n";
    }

    bool uuidexistswarning = false;
    bool dataexistswarning = false;
    bool datatypewarning = false;

    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            uuidexistswarning=true;
            continue;
        }
        for( const auto &label : column_format ) {
            if( !doesPrimitiveDataExist(UUID,label.c_str()) ){
                dataexistswarning=true;
                file << 0 << " ";
                continue;
            }
            HeliosDataType type = getPrimitiveDataType( UUID, label.c_str() );
            if( type == HELIOS_TYPE_INT ){
                int data;
                getPrimitiveData( UUID, label.c_str(), data );
                file << data << " ";
            }else if( type == HELIOS_TYPE_UINT ) {
                uint data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            }else if( type == HELIOS_TYPE_FLOAT ) {
                float data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            }else if( type == HELIOS_TYPE_DOUBLE ) {
                double data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            }else if( type == HELIOS_TYPE_STRING ) {
                std::string data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            }else{
                datatypewarning=true;
                file << 0 << " ";
            }
        }
        file.seekp(-1, std::ios_base::end);
        file << "\n";
    }

    if( uuidexistswarning ){
        std::cerr << "WARNING (Context::writePrimitiveData): Vector of UUIDs passed to writePrimitiveData() function contained UUIDs that do not exist, which were skipped." << std::endl;
    }
    if( dataexistswarning ){
        std::cerr << "WARNING (Context::writePrimitiveData): Primitive data requested did not exist for one or more primitives. A default value of 0 was written in these cases." << std::endl;
    }
    if( datatypewarning ){
        std::cerr << "WARNING (Context::writePrimitiveData): Only scalar primitive data types (uint, int, float, and double) are supported for this function. A column of 0's was written in these cases." << std::endl;
    }

    file.close();

}



Context::~Context(){

    for(auto & primitive : primitives){
        Primitive* prim = getPrimitivePointer_private(primitive.first);
        delete prim;
    }

    for(auto & object : objects){
        CompoundObject* obj = getObjectPointer(object.first);
        delete obj;
    }

}

PrimitiveType Context::getPrimitiveType(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getType();
}

void Context::setPrimitiveParentObjectID(uint UUID, uint objID){
    getPrimitivePointer_private(UUID)->setParentObjectID(objID);
}

void Context::setPrimitiveParentObjectID(const std::vector<uint> &UUIDs, uint objID) {
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->setParentObjectID(objID);
    }
}

uint Context::getPrimitiveParentObjectID(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getParentObjectID();
}

float Context::getPrimitiveArea(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getArea();
}

helios::vec3 Context::getPrimitiveNormal(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getNormal();
}

void Context::getPrimitiveTransformationMatrix(uint UUID, float (&T)[16] ) const {
    getPrimitivePointer_private(UUID)->getTransformationMatrix( T );
}

void Context::setPrimitiveTransformationMatrix(uint UUID, float (&T)[16] ) {
    getPrimitivePointer_private(UUID)->setTransformationMatrix(T);
}

void Context::setPrimitiveTransformationMatrix(const std::vector<uint> &UUIDs, float (&T)[16] ) {
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->setTransformationMatrix(T);
    }
}

std::vector<helios::vec3> Context::getPrimitiveVertices(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getVertices();
}


helios::RGBcolor Context::getPrimitiveColor(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColor();
}

helios::RGBcolor Context::getPrimitiveColorRGB(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColorRGB();
}

helios::RGBAcolor Context::getPrimitiveColorRGBA(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getColorRGBA();
}

void Context::setPrimitiveColor(uint UUID, const RGBcolor &color) {
    getPrimitivePointer_private(UUID)->setColor( color );
}

void Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBcolor &color) {
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->setColor(color);
    }
}

void Context::setPrimitiveColor(uint UUID, const RGBAcolor &color) {
    getPrimitivePointer_private(UUID)->setColor( color );
}

void Context::setPrimitiveColor(const std::vector<uint> &UUIDs, const RGBAcolor &color) {
    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->setColor(color);
    }
}

std::string Context::getPrimitiveTextureFile( uint UUID ) const{
    return getPrimitivePointer_private(UUID)->getTextureFile();
}

helios::int2 Context::getPrimitiveTextureSize( uint UUID ) const{
    std::string texturefile = getPrimitivePointer_private(UUID)->getTextureFile();
    if( !texturefile.empty() && textures.find(texturefile)!=textures.end() ){
        return textures.at(texturefile).getSize();
    }
    return make_int2(0,0);
}

std::vector<helios::vec2> Context::getPrimitiveTextureUV( uint UUID ) const{
    return getPrimitivePointer_private(UUID)->getTextureUV();
}

bool Context::primitiveTextureHasTransparencyChannel( uint UUID ) const{
    std::string texturefile = getPrimitivePointer_private(UUID)->getTextureFile();
    if( !texturefile.empty() && textures.find(texturefile)!=textures.end() ){
        return textures.at(texturefile).hasTransparencyChannel();
    }
    return false;
}

const std::vector<std::vector<bool>> * Context::getPrimitiveTextureTransparencyData(uint UUID ) const{
    if(primitiveTextureHasTransparencyChannel(UUID) ){
        const std::vector<std::vector<bool> > *data = textures.at(getPrimitivePointer_private(UUID)->getTextureFile()).getTransparencyData();
        return data;
    }else{
        throw( std::runtime_error("ERROR (getPrimitiveTransparencyData): Texture transparency data does not exist for primitive " + std::to_string(UUID) + ".") );
    }
}

void Context::overridePrimitiveTextureColor( uint UUID ){
    getPrimitivePointer_private(UUID)->overrideTextureColor();
};

void Context::usePrimitiveTextureColor( uint UUID ){
    getPrimitivePointer_private(UUID)->useTextureColor();
};

bool Context::isPrimitiveTextureColorOverridden( uint UUID ) const{
    return getPrimitivePointer_private(UUID)->isTextureColorOverridden();
}

float Context::getPrimitiveSolidFraction( uint UUID ) const{
    return getPrimitivePointer_private(UUID)->getSolidFraction();
}

void Context::printPrimitiveInfo(uint UUID) const{

    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Info for UUID " << UUID << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    PrimitiveType type = getPrimitiveType(UUID);
    std::string stype;
    if( type == 0){
        stype =  "PRIMITIVE_TYPE_PATCH";
    }else if(type == 1){
        stype =  "PRIMITIVE_TYPE_TRIANGLE";
    }else if(type == 2){
        stype =  "PRIMITIVE_TYPE_VOXEL";
    }

    std::cout << "Type: " << stype << std::endl;
    std::cout << "Surface Area: " << getPrimitiveArea(UUID) << std::endl;
    std::cout << "Normal Vector: " << getPrimitiveNormal(UUID) << std::endl;

    if(type == PRIMITIVE_TYPE_PATCH)
    {
        std::cout << "Patch Center: " << getPatchCenter(UUID) << std::endl;
        std::cout << "Patch Size: " << getPatchSize(UUID) << std::endl;

    }else if(type == PRIMITIVE_TYPE_VOXEL){

        std::cout << "Voxel Center: " << getVoxelCenter(UUID) << std::endl;
        std::cout << "Voxel Size: " << getVoxelSize(UUID) << std::endl;
    }

    std::vector<vec3> primitive_vertices = getPrimitiveVertices(UUID);
    std::cout << "Vertices: " << std::endl;
    for(uint i=0; i<primitive_vertices.size();i++)
    {
        std::cout << "   " << primitive_vertices.at(i) << std::endl;
    }

    float T[16];
    getPrimitiveTransformationMatrix(UUID, T);
    std::cout << "Transform: " << std::endl;
    std::cout << "   " << T[0] << "      " << T[1] << "      " << T[2] << "      " << T[3] << std::endl;
    std::cout << "   " << T[4] << "      " << T[5] << "      " << T[6] << "      " << T[7] << std::endl;
    std::cout << "   " << T[8] << "      " << T[9] << "      " << T[10] << "      " << T[11] << std::endl;
    std::cout << "   " << T[12] << "      " << T[13] << "      " << T[14] << "      " << T[15] << std::endl;

    std::cout << "Color: " << getPrimitiveColor(UUID) << std::endl;
    std::cout << "Texture File: " << getPrimitiveTextureFile(UUID) << std::endl;
    std::cout << "Texture Size: " << getPrimitiveTextureSize(UUID) << std::endl;
    std::cout << "Texture UV: " << std::endl;
    std::vector<vec2> uv = getPrimitiveTextureUV(UUID);
    for(uint i=0; i<uv.size();i++)
    {
        std::cout << "   " << uv.at(i) << std::endl;
    }

    std::cout << "Texture Transparency: " << primitiveTextureHasTransparencyChannel(UUID) << std::endl;
    std::cout << "Color Overridden: " << isPrimitiveTextureColorOverridden(UUID) << std::endl;
    std::cout << "Solid Fraction: " << getPrimitiveSolidFraction(UUID) << std::endl;


    std::cout << "Primitive Data: " << std::endl;
    Primitive* pointer = getPrimitivePointer_private(UUID);
    std::vector<std::string> pd = pointer->listPrimitiveData();
    for(uint i=0; i<pd.size();i++)
    {
        uint dsize = getPrimitiveDataSize(UUID, pd.at(i).c_str());
        HeliosDataType dtype = getPrimitiveDataType(UUID, pd.at(i).c_str());
        std::string dstype;

        if( dtype==HELIOS_TYPE_INT ){
            dstype = "HELIOS_TYPE_INT";
        }else if( dtype==HELIOS_TYPE_UINT ){
            dstype = "HELIOS_TYPE_UINT";
        }else if( dtype==HELIOS_TYPE_FLOAT ){
            dstype = "HELIOS_TYPE_FLOAT";
        }else if( dtype==HELIOS_TYPE_DOUBLE ){
            dstype = "HELIOS_TYPE_DOUBLE";
        }else if( dtype==HELIOS_TYPE_VEC2 ){
            dstype = "HELIOS_TYPE_VEC2";
        }else if( dtype==HELIOS_TYPE_VEC3 ){
            dstype = "HELIOS_TYPE_VEC3";
        }else if( dtype==HELIOS_TYPE_VEC4 ){
            dstype = "HELIOS_TYPE_VEC4";
        }else if( dtype==HELIOS_TYPE_INT2 ){
            dstype = "HELIOS_TYPE_INT2";
        }else if( dtype==HELIOS_TYPE_INT3 ){
            dstype = "HELIOS_TYPE_INT3";
        }else if( dtype==HELIOS_TYPE_INT4 ){
            dstype = "HELIOS_TYPE_INT4";
        }else if( dtype==HELIOS_TYPE_STRING ){
            dstype = "HELIOS_TYPE_STRING";
        }else{
            assert(false);
        }


        std::cout << "   " << "[name: " << pd.at(i) << ", type: " << dstype << ", size: " << dsize << "]:" << std::endl;


        if( dtype==HELIOS_TYPE_INT ){
            std::vector<int> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_UINT ){
            std::vector<uint> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_FLOAT ){
            std::vector<float> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_DOUBLE ){
            std::vector<double> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_VEC2 ){
            std::vector<vec2> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_VEC3 ){
            std::vector<vec3> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_VEC4 ){
            std::vector<vec4> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_INT2 ){
            std::vector<int2> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_INT3 ){
            std::vector<int3> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_INT4 ){
            std::vector<int4> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else if( dtype==HELIOS_TYPE_STRING ){
            std::vector<std::string> pdata;
            getPrimitiveData( UUID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    break;
                }

            }
        }else{
            assert(false);
        }

    }
    std::cout << "-------------------------------------------" << std::endl;
}