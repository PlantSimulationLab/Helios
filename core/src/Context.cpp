/** \file "Context.cpp" Context declarations.

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

Context::Context(){

    //---- ALL DEFAULT VALUES ARE SET HERE ----//

    iscontextinitialized=true;

    sim_date = make_Date(1,6,2000);

    sim_time = make_Time(12,0);

    // --- Initialize random number generator ---- //

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    generator.seed(seed);

    // --- Set Geometry as `Clean' --- //

    isgeometrydirty = false;

    currentUUID = 0;

    currentObjectID = 1; //object ID of 0 is reserved for default object

}

void Context::seedRandomGenerator(uint seed){
    generator.seed(seed);
}

std::minstd_rand0* Context::getRandomGenerator(){
    return &generator;
}

void Context::addTexture( const char* texture_file ){
    if( textures.find(texture_file)==textures.end() ){//texture has not already been added
        Texture text( texture_file );
        textures[ texture_file ] = text;
    }
}

bool Context::doesTextureFileExist(const char* texture_file ) const{
    if (FILE *file = fopen(texture_file, "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }
}

Texture::Texture( const char* texture_file ){
    filename = texture_file;

    //------ determine if transparency channel exists ---------//

    //check if texture file has extension ".png"
    std::string ext = getFileExtension(filename);
    if( ext!=".png" ){
        hastransparencychannel = false;
    }else{
        hastransparencychannel = PNGHasAlpha( filename.c_str() );
    }

    //-------- load transparency channel (if exists) ------------//

    if( hastransparencychannel ){
        transparencydata = readPNGAlpha( filename );
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

void Primitive::setTextureFile( const char* texture ){
  texturefile = texture;
}

std::vector<vec2> Primitive::getTextureUV(){
    return uv;
}

void Primitive::setTextureUV( const std::vector<vec2> &a_uv ){
  uv = a_uv;
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

void Primitive::setSolidFraction( float solidFraction ){
  solid_fraction = solidFraction;
}

void Patch::calculateSolidFraction( const std::map<std::string,Texture> &textures ){

  if( textures.at(texturefile).hasTransparencyChannel() ){
    const std::vector<std::vector<bool> >* alpha = textures.at(texturefile).getTransparencyData();
    int A = 0;
    int At = 0;
    int2 sz = textures.at(texturefile).getSize();
    int2 uv_min( std::max(0,(int)roundf(uv.at(0).x*float(sz.x))), std::max(0,(int)roundf((1.f-uv.at(2).y)*float(sz.y))) );
    int2 uv_max( std::min(sz.x-1,(int)roundf(uv.at(2).x*float(sz.x))), std::min(sz.y-1,(int)roundf((1.f-uv.at(0).y)*float(sz.y))) );
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

}

void Triangle::calculateSolidFraction( const std::map<std::string,Texture> &textures ){

  if( textures.at(texturefile).hasTransparencyChannel() ){
    const std::vector<std::vector<bool> >* alpha = textures.at(texturefile).getTransparencyData();
    int2 sz = textures.at(texturefile).getSize();
    int2 uv_min( std::max(0,(int)round(fminf(fminf(uv.at(0).x,uv.at(1).x),uv.at(2).x)*float(sz.x))), std::max(0,(int)round(fmin(fminf(uv.at(0).y,uv.at(1).y),uv.at(2).y)*float(sz.y))) );
    int2 uv_max( std::min(sz.x-1,(int)round(fmaxf(fmaxf(uv.at(0).x,uv.at(1).x),uv.at(2).x)*float(sz.x))), std::min(sz.y-1,(int)round(fmaxf(fmaxf(uv.at(0).y,uv.at(1).y),uv.at(2).y)*float(sz.y))) );
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
          if( alpha->at(alpha->size()-j-1 ).at(i) ){
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

}

void Voxel::calculateSolidFraction( const std::map<std::string,Texture> &textures ){

}

bool Triangle::edgeFunction(const helios::vec2 &a, const helios::vec2 &b, const helios::vec2 &c){
  return ((c.y - a.y) * (b.x - a.x)-(c.x - a.x) * (b.y - a.y) >= 0);
}

void Primitive::applyTransform( float (&T)[16] ){
    if( parent_object_ID!=0 ){
        std::cout << "WARNING (Primitive::applyTransform): Cannot transform individual primitives within a compound object. Use the setter function for objects." << std::endl;
        return;
    }

    matmult(T,transform,transform);
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
        helios_runtime_error( "ERROR (Patch::rotate): Rotation axis should be one of x, y, or z." );
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
        helios_runtime_error( "ERROR (Triangle::rotate): Rotation axis should be one of x, y, or z." );
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

Patch::Patch( const RGBAcolor& a_color, uint a_parent_objID, uint a_UUID ){

    makeIdentityMatrix( transform );

    color = a_color;
    assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    solid_fraction = 1.f;
    texturefile = "";
    texturecoloroverridden = false;

}

Patch::Patch( const char* a_texturefile, float a_solid_fraction, uint a_parent_objID, uint a_UUID ){

    makeIdentityMatrix( transform );

    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;
    texturefile = a_texturefile;
    solid_fraction = a_solid_fraction;
    texturecoloroverridden = false;

}

Patch::Patch( const char* a_texturefile, const std::vector<vec2>& a_uv, const std::map<std::string,Texture> &textures, uint a_parent_objID, uint a_UUID ){

    makeIdentityMatrix( transform );

    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_PATCH;

    texturefile = a_texturefile;
    uv = a_uv;
    texturecoloroverridden = false;

    if( uv.size()==4 && uv.at(0).x==0 && uv.at(0).y==0 && uv.at(1).x==1 && uv.at(1).y==0 && uv.at(2).x==1 && uv.at(2).y==1 && uv.at(3).x== 0 && uv.at(3).y==1 ){
      solid_fraction = textures.at(texturefile).getSolidFraction();
    }else {
      this->calculateSolidFraction(textures);
    }

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

Triangle::Triangle(  const vec3& a_vertex0, const vec3& a_vertex1, const vec3& a_vertex2, const RGBAcolor& a_color, uint a_parent_objID, uint a_UUID ){

    makeTransformationMatrix(a_vertex0,a_vertex1,a_vertex2);
    color = a_color;
    parent_object_ID = a_parent_objID;
    UUID = a_UUID;
    prim_type = PRIMITIVE_TYPE_TRIANGLE;
    texturefile = "";
    solid_fraction = 1.f;
    texturecoloroverridden = false;

}

Triangle::Triangle( const vec3& a_vertex0, const vec3& a_vertex1, const vec3& a_vertex2, const char* a_texturefile, const std::vector<vec2>& a_uv, float solid_fraction, uint a_parent_objID, uint a_UUID ){

  makeTransformationMatrix(a_vertex0,a_vertex1,a_vertex2);
  color = make_RGBAcolor(RGB::red,1);
  parent_object_ID = a_parent_objID;
  UUID = a_UUID;
  prim_type = PRIMITIVE_TYPE_TRIANGLE;

  texturefile = a_texturefile;
  uv = a_uv;
  solid_fraction = solid_fraction;
  texturecoloroverridden = false;

}

Triangle::Triangle( const vec3& a_vertex0, const vec3& a_vertex1, const vec3& a_vertex2, const char* a_texturefile, const std::vector<vec2>& a_uv, const std::map<std::string,Texture> &textures, uint a_parent_objID, uint a_UUID ){

  makeTransformationMatrix(a_vertex0,a_vertex1,a_vertex2);
  color = make_RGBAcolor(RGB::red,1);
  parent_object_ID = a_parent_objID;
  UUID = a_UUID;
  prim_type = PRIMITIVE_TYPE_TRIANGLE;

  texturefile = a_texturefile;
  uv = a_uv;
  solid_fraction = 1.f;
  texturecoloroverridden = false;

  this->calculateSolidFraction(textures);

}

vec3 Triangle::getVertex( int number ){

    if( number<0 || number>2 ){
        helios_runtime_error("ERROR (getVertex): vertex index must be 1, 2, or 3.");
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

Voxel::Voxel( const RGBAcolor& a_color, uint a_parent_objID, uint a_UUID ){

    makeIdentityMatrix(transform);

    color = a_color;
    assert( color.r>=0 && color.r<=1 && color.g>=0 && color.g<=1 && color.b>=0 && color.b<=1 );
    solid_fraction = 1.f;
    parent_object_ID = a_parent_objID;
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
        helios_runtime_error("ERROR (setDate): Day of month is out of range (day of " + std::to_string(day) + " was given).");
    }else if( month<1 || month>12){
        helios_runtime_error("ERROR (setDate): Month of year is out of range (month of " + std::to_string(month) + " was given).");
    }else if( year<1000 ){
        helios_runtime_error("ERROR (setDate): Year should be specified in YYYY format.");
    }

    sim_date = make_Date(day,month,year);

}

void Context::setDate( Date date ){

    if( date.day<1 || date.day>31 ){
        helios_runtime_error("ERROR (setDate): Day of month is out of range (day of " + std::to_string(date.day) + " was given).");
    }else if( date.month<1 || date.month>12){
        helios_runtime_error("ERROR (setDate): Month of year is out of range (month of " + std::to_string(date.month) + " was given).");
    }else if( date.year<1000 ){
        helios_runtime_error("ERROR (setDate): Year should be specified in YYYY format.");
    }

    sim_date = date;

}

void Context::setDate( int Julian_day, int year ){

    if( Julian_day<1 || Julian_day>366 ){
        helios_runtime_error("ERROR (setDate): Julian day out of range.");
    }else if( year<1000 ){
        helios_runtime_error("ERROR (setDate): Year should be specified in YYYY format.");
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
        helios_runtime_error("ERROR (setTime): Second out of range (0-59).");
    }else if( minute<0 || minute>59 ){
        helios_runtime_error("ERROR (setTime): Minute out of range (0-59).");
    }else if( hour<0 || hour>23 ){
        helios_runtime_error("ERROR (setTime): Hour out of range (0-23).");
    }

    sim_time = make_Time(hour,minute,second);

}

void Context::setTime( Time time ){

    if( time.minute<0 || time.minute>59 ){
        helios_runtime_error("ERROR (setTime): Minute out of range (0-59).");
    }else if( time.hour<0 || time.hour>23 ){
        helios_runtime_error("ERROR (setTime): Hour out of range (0-23).");
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
        helios_runtime_error("ERROR (randu): Maximum value of range must be greater than minimum value of range.");
        return 0;
    }else if( maxrange==minrange ){
        return minrange;
    }else{
        return minrange+unif_distribution(generator)*(maxrange-minrange);
    }
}

int Context::randu( int minrange, int maxrange ){
    if( maxrange<minrange ){
        helios_runtime_error("ERROR (randu): Maximum value of range must be greater than minimum value of range.");
        return 0;
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
        helios_runtime_error("ERROR (addPatch): Size of patch must be greater than 0.");
    }

    auto* patch_new = (new Patch( color, 0, currentUUID ));

//    if( patch_new->getArea()==0 ){
//        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
//    }

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

    //texture must have type PNG or JPEG
    std::string fn = texture_file;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addPatch): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texture_file) ){
        helios_runtime_error("ERROR (Context::addPatch): Texture file " + std::string(texture_file) + " does not exist.");
    }

    addTexture( texture_file );

    auto* patch_new = (new Patch( texture_file, textures.at(texture_file).getSolidFraction(), 0, currentUUID ));

//    if( patch_new->getArea()==0 ){
//        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
//    }

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

    //texture must have type PNG or JPEG
    std::string fn = texture_file;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addPatch): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texture_file) ){
        helios_runtime_error("ERROR (Context::addPatch): Texture file " + std::string(texture_file) + " does not exist.");
    }

    if( size.x==0 || size.y==0 ){
        helios_runtime_error("ERROR (addPatch): Size of patch must be greater than 0.");
    }

    if( uv_center.x-0.5*uv_size.x<-1e-3 || uv_center.y-0.5*uv_size.y<-1e-3 || uv_center.x+0.5*uv_size.x-1.f>1e-3 || uv_center.y+0.5*uv_size.y-1.f>1e-3 ){
        helios_runtime_error("ERROR (addPatch): Invalid texture coordinates. uv_center-0.5*uv_size should be >=0 and uv_center+0.5*uv_size should be <=1.");
    }

    addTexture( texture_file );

    std::vector<helios::vec2> uv;
    uv.resize(4);
    uv.at(0) = uv_center+make_vec2(-0.5f*uv_size.x,-0.5f*uv_size.y);
    uv.at(1) = uv_center+make_vec2(+0.5f*uv_size.x,-0.5f*uv_size.y);
    uv.at(2) =  uv_center+make_vec2(+0.5f*uv_size.x,+0.5f*uv_size.y);
    uv.at(3) =  uv_center+make_vec2(-0.5f*uv_size.x,+0.5f*uv_size.y);

    auto* patch_new = (new Patch( texture_file, uv, textures, 0, currentUUID ));

//    if( patch_new->getArea()==0 ){
//        helios_runtime_error("ERROR (Context::addPatch): Patch has area of zero.");
//    }

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

uint Context::addTriangle( const vec3& vertex0, const vec3& vertex1, const vec3& vertex2, const RGBAcolor& color ){

    auto* tri_new = (new Triangle( vertex0, vertex1, vertex2, color, 0, currentUUID ));

//    if( tri_new->getArea()==0 ){
//        helios_runtime_error("ERROR (Context::addTriangle): Triangle has area of zero.");
//    }

    primitives[currentUUID] = tri_new;
    markGeometryDirty();
    currentUUID++;
    return currentUUID-1;
}

uint Context::addTriangle( const helios::vec3& vertex0, const helios::vec3& vertex1, const helios::vec3& vertex2, const char* texture_file, const helios::vec2& uv0, const helios::vec2& uv1, const helios::vec2& uv2 ){

    //texture must have type PNG or JPEG
    std::string fn = texture_file;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addTriangle): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texture_file) ){
        helios_runtime_error("ERROR (Context::addTriangle): Texture file " + std::string(texture_file) + " does not exist.");
    }

    addTexture( texture_file );

    std::vector<helios::vec2> uv;
    uv.resize(3);
    uv.at(0) = uv0;
    uv.at(1) = uv1;
    uv.at(2) = uv2;

    auto* tri_new = (new Triangle( vertex0, vertex1, vertex2, texture_file, uv, textures, 0, currentUUID ));

//    if( tri_new->getArea()==0 ){
//        helios_runtime_error("ERROR (Context::addTriangle): Triangle has area of zero.");
//    }

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

    auto* voxel_new = (new Voxel( color, 0, currentUUID ));

    if( size.x*size.y*size.z==0 ){
        helios_runtime_error("ERROR (Context::addVoxel): Voxel has size of zero.");
    }

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

    float T[16];
    makeTranslationMatrix(shift,T);

    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::rotatePrimitive(uint UUID, float rot, const char* axis ){
    getPrimitivePointer_private(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive( const std::vector<uint>& UUIDs, float rot, const char* axis ){

    float T[16];
    if( strcmp(axis,"z")==0 ){
        makeRotationMatrix(rot,"z",T);
    }else if( strcmp(axis,"y")==0 ){
        makeRotationMatrix(rot,"y",T);
    }else if( strcmp(axis,"x")==0 ){
        makeRotationMatrix(rot,"x",T);
    }else{
        helios_runtime_error( "ERROR (Context::rotatePrimitive): Rotation axis should be one of x, y, or z." );
    }

    for( uint UUID : UUIDs){
        if( strcmp(axis,"z")!=0 && getPrimitivePointer_private(UUID)->getType()==PRIMITIVE_TYPE_VOXEL ){
            std::cout << "WARNING (Context::rotatePrimitive): Voxels can only be rotate about the z-axis. Ignoring this rotation." << std::endl;
        }
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::rotatePrimitive(uint UUID, float rot, const helios::vec3& axis ){
    getPrimitivePointer_private(UUID)->rotate(rot,axis);
}

void Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const vec3 &axis ){

    float T[16];
    makeRotationMatrix(rot,axis,T);

    for( uint UUID : UUIDs){
        if( getPrimitivePointer_private(UUID)->getType()==PRIMITIVE_TYPE_VOXEL ){
            std::cout << "WARNING (Context::rotatePrimitive): Voxels can only be rotate about the z-axis. Ignoring this rotation." << std::endl;
        }
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::rotatePrimitive( uint UUID, float rot, const helios::vec3& origin, const helios::vec3& axis ){
    getPrimitivePointer_private(UUID)->rotate(rot,origin,axis);
}

void Context::rotatePrimitive(const std::vector<uint>& UUIDs, float rot, const helios::vec3& origin, const vec3 &axis ){

    float T[16];
    makeRotationMatrix(rot,origin,axis,T);

    for( uint UUID : UUIDs){
        if( getPrimitivePointer_private(UUID)->getType()==PRIMITIVE_TYPE_VOXEL ){
            std::cout << "WARNING (Context::rotatePrimitive): Voxels can only be rotate about the z-axis. Ignoring this rotation." << std::endl;
        }
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::scalePrimitive(uint UUID, const helios::vec3& S ){
    getPrimitivePointer_private(UUID)->scale(S);
}

void Context::scalePrimitive( const std::vector<uint>& UUIDs, const helios::vec3& S ){

    float T[16];
    makeScaleMatrix(S,T);

    for( uint UUID : UUIDs){
        getPrimitivePointer_private(UUID)->applyTransform(T);
    }
}

void Context::deletePrimitive( const std::vector<uint>& UUIDs ){
    for( uint UUID : UUIDs){
        deletePrimitive( UUID );
    }
}

void Context::deletePrimitive(uint UUID ){
    
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (deletePrimitive): UUID of " + std::to_string(UUID) + " not found in the context.");
    }
    
    Primitive* prim = primitives.at(UUID);
    
    if( prim->getParentObjectID()!=0 ){//primitive belongs to an object
        
        uint ObjID = prim->getParentObjectID();
        if( doesObjectExist(ObjID) ) {
            objects.at(ObjID)->deleteChildPrimitive(UUID);
            if(getObjectPointer_private(ObjID)->getPrimitiveUUIDs().empty() )
            {
                CompoundObject* obj = objects.at(ObjID);
                delete obj;
                objects.erase(ObjID);
            }
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
        helios_runtime_error("ERROR (copyPrimitive): UUID of " + std::to_string(UUID) + " not found in the context.");
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
            patch_new = (new Patch( p->getColorRGBA(), parentID, currentUUID ));
        }else{
            std::string texture_file = p->getTextureFile();
            if( uv.size()==4 ){
                patch_new = (new Patch( texture_file.c_str(), solid_fraction, parentID, currentUUID ));
                patch_new->setTextureUV(uv);
            }else{
                patch_new = (new Patch( texture_file.c_str(), solid_fraction, parentID, currentUUID ));
            }
        }
        float transform[16];
        p->getTransformationMatrix(transform);
        patch_new->setTransformationMatrix(transform);
        primitives[currentUUID] = patch_new;
    }else if( type==PRIMITIVE_TYPE_TRIANGLE ){
        Triangle* p = getTrianglePointer_private(UUID);
        std::vector<vec3> vertices = p->getVertices();
        std::vector<vec2> uv = p->getTextureUV();
        Triangle* tri_new;
        if( !p->hasTexture() ){
            tri_new = (new Triangle( vertices.at(0), vertices.at(1), vertices.at(2), p->getColorRGBA(), parentID, currentUUID ));
        }else{
            std::string texture_file = p->getTextureFile();
            float solid_fraction = p->getArea()/calculateTriangleArea( vertices.at(0), vertices.at(1), vertices.at(2) );
            tri_new = (new Triangle( vertices.at(0), vertices.at(1), vertices.at(2), texture_file.c_str(), uv, solid_fraction, parentID, currentUUID ));
            tri_new->setSolidFraction(solid_fraction);
        }
        float transform[16];
        p->getTransformationMatrix(transform);
        tri_new->setTransformationMatrix(transform);
        primitives[currentUUID] = tri_new;
    }else if( type==PRIMITIVE_TYPE_VOXEL ){
        Voxel* p = getVoxelPointer_private(UUID);
        Voxel* voxel_new;
        //if( !p->hasTexture() ){
        voxel_new = (new Voxel( p->getColorRGBA(), parentID, currentUUID ));
        //}else{
        //  voxel_new = (new Voxel( p->getColorRGBA(), currentUUID ));
        /* \todo Texture-mapped voxels constructor here */
        //}
        float transform[16];
        p->getTransformationMatrix(transform);
        voxel_new->setTransformationMatrix(transform);
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

Primitive* Context::getPrimitivePointer( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getPrimitivePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
    return primitives.at(UUID);
}

Primitive* Context::getPrimitivePointer_private( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getPrimitivePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
    return primitives.at(UUID);
}

bool Context::doesPrimitiveExist(uint UUID ) const{
    return primitives.find(UUID) != primitives.end();
}

bool Context::doesPrimitiveExist( const std::vector<uint> &UUIDs ) const{
    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            return false;
        }
    }
    return true;
}

Patch* Context::getPatchPointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        helios_runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
    return dynamic_cast<Patch*>(primitives.at(UUID));
}

Patch* Context::getPatchPointer_private(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        helios_runtime_error("ERROR (getPatchPointer): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
    return dynamic_cast<Patch*>(primitives.at(UUID));
}

helios::vec2 Context::getPatchSize( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getPatchSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        helios_runtime_error("ERROR (getPatchSize): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
    return dynamic_cast<Patch*>(primitives.at(UUID))->getSize();
}

helios::vec3 Context::getPatchCenter( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getPatchCenter): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_PATCH ){
        helios_runtime_error("ERROR (getPatchCenter): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
    return dynamic_cast<Patch*>(primitives.at(UUID))->getCenter();
}

Triangle* Context::getTrianglePointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        helios_runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " is not a triangle.");
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID));
}

Triangle* Context::getTrianglePointer_private(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        helios_runtime_error("ERROR (getTrianglePointer): UUID of " + std::to_string(UUID) + " is not a triangle.");
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID));
}

helios::vec3 Context::getTriangleVertex( uint UUID, uint number ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getTriangleVertex): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_TRIANGLE ){
        helios_runtime_error("ERROR (getTriangleVertex): UUID of " + std::to_string(UUID) + " is not a triangle.");
    }else if( number>2 ){
        helios_runtime_error("ERROR (getTriangleVertex): Vertex index must be one of 0, 1, or 2.");
    }
    return dynamic_cast<Triangle*>(primitives.at(UUID))->getVertex( number );
}

Voxel* Context::getVoxelPointer(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        helios_runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " is not a voxel.");
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID));
}

Voxel* Context::getVoxelPointer_private(uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        helios_runtime_error("ERROR (getVoxelPointer): UUID of " + std::to_string(UUID) + " is not a voxel.");
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID));
}

helios::vec3 Context::getVoxelSize( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getVoxelSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        helios_runtime_error("ERROR (getVoxelSize): UUID of " + std::to_string(UUID) + " is not a patch.");
    }
    return dynamic_cast<Voxel*>(primitives.at(UUID))->getSize();
}

helios::vec3 Context::getVoxelCenter( uint UUID ) const{
    if( primitives.find(UUID) == primitives.end() ){
        helios_runtime_error("ERROR (getVoxelCenter): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }else if( primitives.at(UUID)->getType()!=PRIMITIVE_TYPE_VOXEL ){
        helios_runtime_error("ERROR (getVoxelCenter): UUID of " + std::to_string(UUID) + " is not a patch.");
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
                    std::cout << "WARNING (Context::addTimeseriesData): Skipping duplicate timeseries date/time." << std::endl;
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

    helios_runtime_error("ERROR (Context::addTimeseriesData): Failed to insert timeseries data for unknown reason.");

}

void Context::setCurrentTimeseriesPoint(const char* label, uint index ){
    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        helios_runtime_error("ERROR (setCurrentTimeseriesPoint): Timeseries variable `" + std::string(label) + "' does not exist.");
    }
    setDate( queryTimeseriesDate( label, index ) );
    setTime( queryTimeseriesTime( label, index ) );
}

float Context::queryTimeseriesData(const char* label, const Date &date, const Time &time ) const{
    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        helios_runtime_error("ERROR (setCurrentTimeseriesData): Timeseries variable `" + std::string(label) + "' does not exist.");
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
            helios_runtime_error("ERROR (queryTimeseriesData): Failed to query timeseries data for unknown reason.");
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
        helios_runtime_error("ERROR( Context::getTimeseriesData): Timeseries variable " + std::string(label) + " does not exist.");
    }

    return timeseries_data.at(label).at(index);

}

Time Context::queryTimeseriesTime( const char* label, const uint index ) const{

    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        helios_runtime_error("ERROR( Context::getTimeseriesTime): Timeseries variable " + std::string(label) + " does not exist.");
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
        helios_runtime_error("ERROR( Context::getTimeseriesDate): Timeseries variable " + std::string(label) + " does not exist.");
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

    uint size = 0;
    if( timeseries_data.find(label) == timeseries_data.end() ){ //does not exist
        helios_runtime_error("ERROR (Context::getTimeseriesDate): Timeseries variable `" + std::string(label) + "' does not exist.");
    }else{
        size = timeseries_data.at(label).size();
    }

    return size;
}

bool Context::doesTimeseriesVariableExist( const char* label ) const{

    if( timeseries_data.find(label) == timeseries_data.end() ) { //does not exist
        return false;
    }else{
        return true;
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
                deletePrimitive(p);
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
                deletePrimitive(p);
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
                deletePrimitive(p);
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
                deletePrimitive(UUID);
                delete_count++;
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
}

void CompoundObject::setColor( const helios::RGBAcolor& a_color ){
    for( uint UUID : UUIDs){

        if( context->doesPrimitiveExist( UUID ) ){
            context->setPrimitiveColor( UUID, a_color );
        }

    }
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
        helios_runtime_error("ERROR (CompoundObject::rotate): Rotation axis should be one of x, y, or z.");
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

void CompoundObject::scale( const helios::vec3 &scale ){

    float T[16], T_prim[16];
    makeScaleMatrix( scale, T);
    matmult(T,transform,transform);

    for( uint UUID : UUIDs) {

        if (context->doesPrimitiveExist(UUID)) {

            context->getPrimitiveTransformationMatrix(UUID, T_prim);
            matmult(T, T_prim, T_prim);
            context->setPrimitiveTransformationMatrix(UUID, T_prim);

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

bool Context::areObjectPrimitivesComplete( uint objID ) const{
   return getObjectPointer(objID)->arePrimitivesComplete();
}

CompoundObject* Context::getObjectPointer( uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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
        helios_runtime_error("ERROR (deleteObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
    }
    
    CompoundObject* obj = objects.at(ObjID);
    
    std::vector<uint> UUIDs = obj->getPrimitiveUUIDs();
    
    
    delete obj;
    objects.erase(ObjID);
    
    deletePrimitive(UUIDs);
    
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
        helios_runtime_error("ERROR (copyObject): Object ID of " + std::to_string(ObjID) + " not found in the context.");
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
        int2 subdiv = o->getSubdivisionCount();

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

    copyObjectData( ObjID, currentObjectID );

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
                    if( float(R)<threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,">")==0 ){
                    if( float(R)>threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,"=")==0 ){
                    if( float(R)==threshold ){
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
                    if( float(R)<threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,">")==0 ){
                    if( float(R)>threshold ){
                        output_object_IDs.at(passed_count) = IDs.at(i);
                        passed_count++;
                    }
                }else if( strcmp(comparator,"=")==0 ){
                    if( float(R)==threshold ){
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

void Context::scaleObject( uint ObjID, const helios::vec3 &scalefact ){
    getObjectPointer(ObjID)->scale(scalefact);
}

void Context::scaleObject( const std::vector<uint>& ObjIDs, const helios::vec3 &scalefact ){
    for( uint ID : ObjIDs){
        getObjectPointer(ID)->scale(scalefact);
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
        
        if(!(getObjectPointer(ObjectID)->arePrimitivesComplete())){
            std::cerr << "WARNING (getTileObjectAreaRatio): ObjectID " << ObjectID << " is missing primitives. Area ratio calculated is area of non-missing subpatches divided by the area of an individual subpatch." << std::endl;
        }    
        
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
        }else if(!(getObjectPointer(OBJID)->arePrimitivesComplete())){
            std::cerr << "WARNING (setTileObjectSubdivisionCount): ObjectID " << OBJID << " is missing primitives. Skipping..." << std::endl;
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
        RGBcolor color = getPrimitiveColor(UUIDs_old.front());
        
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
        object_templates.emplace_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.emplace_back(object_primitives);
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
                //copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));
                
                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);
                
                //delete the original object primitives
                deletePrimitive(UUIDs_old);
                
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
        }else if(!(getObjectPointer(OBJID)->arePrimitivesComplete())){
            std::cerr << "WARNING (setTileObjectSubdivisionCount): ObjectID " << OBJID << " is missing primitives. Skipping..." << std::endl;
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
        RGBcolor color = getPrimitiveColor(UUIDs_old.front());
        
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
        object_templates.emplace_back(object_template);
        std::vector<uint> object_primitives = getTileObjectPointer(object_template)->getPrimitiveUUIDs();
        template_primitives.emplace_back(object_primitives);
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
                //copy the template primitives and create a new tile with them
                std::vector<uint> new_primitives = copyPrimitive(template_primitives.at(j));
                
                // change the objectID for the new primitives
                setPrimitiveParentObjectID(new_primitives, textured_tile_ObjectIDs.at(i));
                
                int2 new_subdiv = getTileObjectPointer(object_templates.at(j))->getSubdivisionCount();
                current_object_pointer->setPrimitiveUUIDs(new_primitives);
                current_object_pointer->setSubdivisionCount(new_subdiv);
                
                //delete the original object primitives
                deletePrimitive(UUIDs_old);
                
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
        helios_runtime_error("ERROR (getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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
    
    
    vec3 Y[4];
    Y[0] = make_vec3( -0.5f, -0.5f, 0.f);
    Y[1] = make_vec3( 0.5f, -0.5f, 0.f);
    Y[2] = make_vec3( 0.5f, 0.5f, 0.f);
    Y[3] = make_vec3( -0.5f, 0.5f, 0.f);
    
    for( int i=0; i<4; i++ ){
        vertices[i].x = transform[0] * Y[i].x + transform[1] * Y[i].y + transform[2] * Y[i].z + transform[3];
        vertices[i].y = transform[4] * Y[i].x + transform[5] * Y[i].y + transform[6] * Y[i].z + transform[7];
        vertices[i].z = transform[8] * Y[i].x + transform[9] * Y[i].y + transform[10] * Y[i].z + transform[11];
    }
    
    // vertices.at(0) = context->getPrimitiveVertices( UUIDs.front() ).at(0);
    // vertices.at(1) = context->getPrimitiveVertices( UUIDs.at( subdiv.x-1 ) ).at(1);
    // vertices.at(2) = context->getPrimitiveVertices( UUIDs.at( subdiv.x*subdiv.y-1 ) ).at(2);
    // vertices.at(3) = context->getPrimitiveVertices( UUIDs.at( subdiv.x*subdiv.y-subdiv.x ) ).at(3);
    
    return vertices;
    
}

vec3 Tile::getNormal() const{

    return context->getPrimitiveNormal( UUIDs.front() );

}

std::vector<helios::vec2> Tile::getTextureUV() const{

    std::vector<helios::vec2> uv{ make_vec2(0,0), make_vec2(1,0), make_vec2(1,1), make_vec2(0,1) };

    return uv;

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
        helios_runtime_error("ERROR (getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
    return dynamic_cast<Sphere*>(objects.at(ObjID));
}

helios::vec3 Sphere::getRadius() const{

//    vec3 n0(0,0,0), n1(1,0,0);
//    vec3 n0_T, n1_T;
//
//    vecmult(transform,n0,n0_T);
//    vecmult(transform,n1,n1_T);
//
//    return  (n1_T-n0_T).magnitude();

    vec3 n0(0,0,0);
    vec3 nx(1,0,0);
    vec3 ny(0,1,0);
    vec3 nz(0,0,1);
    vec3 n0_T, nx_T, ny_T, nz_T;

    vecmult(transform,n0,n0_T);
    vecmult(transform,nx,nx_T);
    vecmult(transform,ny,ny_T);
    vecmult(transform,nz,nz_T);

    vec3 radii;
    radii.x = (nx_T-n0_T).magnitude();
    radii.y = (ny_T-n0_T).magnitude();
    radii.z = (nz_T-n0_T).magnitude();

    return radii;

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
        helios_runtime_error("ERROR (getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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
        helios_runtime_error("ERROR (getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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

Disk::Disk(uint a_OID, const std::vector<uint> &a_UUIDs, int2 a_subdiv, const char *a_texturefile,
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
        helios_runtime_error("ERROR (getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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

int2 Disk::getSubdivisionCount() const{
    return subdiv;
}

void Disk::setSubdivisionCount(int2 a_subdiv ){
    subdiv = a_subdiv;
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
        helios_runtime_error("ERROR (getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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
        helios_runtime_error("ERROR (getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
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
        helios_runtime_error("ERROR (Cone::getNode): node number must be 0 or 1.");
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
        helios_runtime_error("ERROR (Cone::getNodeRadius): node number must be 0 or 1.");
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
    return addSphereObject(Ndivs,center,{radius,radius,radius},{0.f,0.75f,0.f}); //Default color is green
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const RGBcolor &color ){
    return addSphereObject(Ndivs,center,{radius,radius,radius},color);
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, float radius, const char* texturefile ){
    return addSphereObject(Ndivs,center,{radius,radius,radius},texturefile);
}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius ){

    return addSphereObject(Ndivs,center,radius,{0.f,0.75f,0.f}); //Default color is green

}

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius, const RGBcolor &color ){

    if( radius.x<=0.f || radius.y<=0.f || radius.z<=0.f ){
        helios_runtime_error("ERROR (addSphereObject): Radius of sphere must be positive.");
    }

    std::vector<uint> UUID;

    float theta;
    float dtheta=float(M_PI)/float(Ndivs);
    float dphi=2.0f*float(M_PI)/float(Ndivs);

    vec3 cart;

    //bottom cap
    for( int j=0; j<Ndivs; j++ ){

        cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI), 0 ) );
        vec3 v0 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+dtheta, float(j)*dphi ) );
        vec3 v1 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);

        UUID.push_back( addTriangle(v0,v1,v2,color) );

    }

    //top cap
    for( int j=0; j<Ndivs; j++ ){

        cart = sphere2cart( make_SphericalCoord(1.f, 0.5f*float(M_PI), 0 ) );
        vec3 v0 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, 0.5f*float(M_PI)-dtheta, float(j)*dphi ) );
        vec3 v1 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, 0.5f*float(M_PI)-dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);

        UUID.push_back( addTriangle(v2,v1,v0,color) );

    }

    //middle
    for( int j=0; j<Ndivs; j++ ){
        for( int i=1; i<Ndivs-1; i++ ){

            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i)*dtheta, float(j)*dphi ) );
            vec3 v0 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v2 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);

            UUID.push_back( addTriangle(v0,v1,v2,color) );
            UUID.push_back( addTriangle(v0,v2,v3,color) );

        }
    }

    auto* sphere_new = (new Sphere(currentObjectID, UUID, Ndivs, "", this));

    float T[16], transform[16];
    sphere_new->getTransformationMatrix( transform );

    makeScaleMatrix(radius,T);
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

uint Context::addSphereObject(uint Ndivs, const vec3 &center, const vec3 &radius, const char* texturefile ){

    if( radius.x<=0.f || radius.y<=0.f || radius.z<=0.f ){
        helios_runtime_error("ERROR (addSphereObject): Radius of sphere must be positive.");
    }

    std::vector<uint> UUID;

    float theta;
    float dtheta=float(M_PI)/float(Ndivs);
    float dphi=2.0f*float(M_PI)/float(Ndivs);

    vec3 cart;

    //bottom cap
    for( int j=0; j<Ndivs; j++ ){

        cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI), 0 ) );
        vec3 v0 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+dtheta, float(j)*dphi ) );
        vec3 v1 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);

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

        cart = sphere2cart( make_SphericalCoord(1.f, 0.5f*float(M_PI), 0 ) );
        vec3 v0 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, 0.5f*float(M_PI)-dtheta, float(j)*dphi ) );
        vec3 v1 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
        cart = sphere2cart( make_SphericalCoord(1.f, 0.5f*float(M_PI)-dtheta, float(j+1)*dphi ) );
        vec3 v2 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);

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

            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i)*dtheta, float(j)*dphi ) );
            vec3 v0 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j)*dphi ) );
            vec3 v1 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i+1)*dtheta, float(j+1)*dphi ) );
            vec3 v2 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);
            cart = sphere2cart( make_SphericalCoord(1.f, -0.5f*float(M_PI)+float(i)*dtheta, float(j+1)*dphi ) );
            vec3 v3 = center + make_vec3(cart.x*radius.x,cart.y*radius.y,cart.z*radius.z);

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

    makeScaleMatrix(radius,T);
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
        helios_runtime_error("ERROR (addTileObject): Size of tile must be greater than 0.");
    }
    if( subdiv.x<1 || subdiv.y<1 ){
        helios_runtime_error("ERROR (addTileObject): Number of tile subdivisions must be greater than 0.");
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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addTileObject): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addTileObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

    if( size.x==0 || size.y==0 ){
        helios_runtime_error("ERROR (addTileObject): Size of tile must be greater than 0.");
    }
    if( subdiv.x<1 || subdiv.y<1 ){
        helios_runtime_error("ERROR (addTileObject): Number of tile subdivisions must be greater than 0.");
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

            auto* patch_new = (new Patch( texturefile, uv,  textures, 0, currentUUID ));

            if( patch_new->getSolidFraction()==0 ){
              delete patch_new;
              continue;
            }

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
        helios_runtime_error("ERROR (Context::addTubeObject): Node and radius arrays are empty.");
    }else if( node_count!=radius.size() ){
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `radius' arguments must agree.");
    }else if( node_count!=color.size() ){
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `color' arguments must agree.");
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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addTubeObject): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addTubeObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

    const uint node_count = nodes.size();

    if( node_count==0 ){
        helios_runtime_error("ERROR (Context::addTubeObject): Node and radius arrays are empty.");
    }else if( node_count!=radius.size() ){
        helios_runtime_error("ERROR (Context::addTubeObject): Size of `nodes' and `radius' arguments must agree.");
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
        helios_runtime_error("ERROR (addBoxObject): Size of box must be positive.");
    }
    if( subdiv.x<1 || subdiv.y<1 || subdiv.z<1 ){
        helios_runtime_error("ERROR (addBoxObject): Number of box subdivisions must be positive.");
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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addBoxObject): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addBoxObject): Texture file " + std::string(texturefile) + " does not exist.");
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
    return addDiskObject(make_int2(Ndivs,1),center,size,make_SphericalCoord(0,0),make_RGBAcolor(1,0,0,1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation ){
    return addDiskObject(make_int2(Ndivs,1),center,size,rotation,make_RGBAcolor(1,0,0,1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color ){
    return addDiskObject(make_int2(Ndivs,1),center,size,rotation,make_RGBAcolor(color,1));
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color ){
    return addDiskObject(make_int2(Ndivs,1),center,size,rotation,color);
}

uint Context::addDiskObject(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){
    return addDiskObject(make_int2(Ndivs,1),center,size,rotation,texture_file);
}

uint Context::addDiskObject(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color ){
    return addDiskObject(Ndivs,center,size,rotation,make_RGBAcolor(color,1));
}

uint Context::addDiskObject(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color ){

    std::vector<uint> UUID;

    UUID.resize(Ndivs.x+Ndivs.x*(Ndivs.y-1)*2);
    int i=0;
    for( int r=0; r < Ndivs.y; r++ ) {
        for (int t = 0; t < Ndivs.x; t++) {

            float dtheta = 2.f * float(M_PI) / float(Ndivs.x);
            float theta = dtheta*float(t);
            float theta_plus = dtheta*float(t+1);

            float rx = size.x/float(Ndivs.y)*float(r);
            float ry = size.y/float(Ndivs.y)*float(r);

            float rx_plus = size.x/float(Ndivs.y)*float(r+1);
            float ry_plus = size.y/float(Ndivs.y)*float(r+1);

            if( r==0 ) {
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            }else{
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0),make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), color);
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
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

uint Context::addDiskObject(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){

    //texture must have type PNG or JPEG
    std::string fn = texture_file;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addDiskObject): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texture_file) ){
        helios_runtime_error("ERROR (Context::addDiskObject): Texture file " + std::string(texture_file) + " does not exist.");
    }

    std::vector<uint> UUID;

    UUID.resize(Ndivs.x+Ndivs.x*(Ndivs.y-1)*2);
    int i=0;
    for( int r=0; r < Ndivs.y; r++ ) {
        for (int t = 0; t < Ndivs.x; t++) {

            float dtheta = 2.f * float(M_PI) / float(Ndivs.x);
            float theta = dtheta*float(t);
            float theta_plus = dtheta*float(t+1);

            float rx = size.x/float(Ndivs.y)*float(r);
            float ry = size.y/float(Ndivs.y)*float(r);
            float rx_plus = size.x/float(Ndivs.y)*float(r+1);
            float ry_plus = size.y/float(Ndivs.y)*float(r+1);

            if( r==0 ) {
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texture_file, make_vec2(0.5,0.5), make_vec2(0.5f*(1.f+cosf(theta)*rx_plus/size.x), 0.5f*(1.f+sinf(theta)*ry_plus/size.y)), make_vec2(0.5f*(1.f+cosf(theta_plus)*rx_plus/size.x), 0.5f*(1.f+sinf(theta_plus)*ry_plus/size.y)));
            }else{
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0),make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), texture_file, make_vec2(0.5f*(1.f+cosf(theta_plus)*rx/size.x), 0.5f*(1.f+sinf(theta_plus)*ry/size.y)), make_vec2(0.5f*(1.f+cosf(theta)*rx/size.x), 0.5f*(1.f+sinf(theta)*ry/size.y)), make_vec2(0.5f*(1.f+cosf(theta)*rx_plus/size.x), 0.5f*(1.f+sinf(theta)*ry_plus/size.y)));
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texture_file, make_vec2(0.5f*(1.f+cosf(theta_plus)*rx/size.x), 0.5f*(1.f+sinf(theta_plus)*ry/size.y)), make_vec2(0.5f*(1.f+cosf(theta)*rx_plus/size.x), 0.5f*(1.f+sinf(theta)*ry_plus/size.y)), make_vec2(0.5f*(1.f+cosf(theta_plus)*rx_plus/size.x), 0.5f*(1.f+sinf(theta_plus)*ry_plus/size.y)));
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
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

    if( !doesPrimitiveExist(UUIDs) ){
        helios_runtime_error("ERROR (Context::addPolymeshObject): One or more of the provided UUIDs does not exist. Cannot create polymesh object.");
    }

    auto* polymesh_new = (new Polymesh(currentObjectID, UUIDs, "", this));

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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addConeObject): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addConeObject): Texture file " + std::string(texturefile) + " does not exist.");
    }

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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addSphereObject): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addSphere): Texture file " + std::string(texturefile) + " does not exist.");
    }

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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addTile): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addTile): Texture file " + std::string(texturefile) + " does not exist.");
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

            auto* patch_new = (new Patch( texturefile, uv, textures, 0, currentUUID ));

            if( patch_new->getSolidFraction()==0 ){
                delete patch_new;
              continue;
            }

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
        helios_runtime_error("ERROR (Context::addTube): Node and radius arrays are empty.");
    }else if( node_count!=radius.size() ){
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.");
    }else if( node_count!=color.size() ){
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `color' arguments must agree.");
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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addTube): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addTube): Texture file " + std::string(texturefile) + " does not exist.");
    }

    const uint node_count = nodes.size();

    if( node_count==0 ){
        helios_runtime_error("ERROR (Context::addTube): Node and radius arrays are empty.");
    }else if( node_count!=radius.size() ){
        helios_runtime_error("ERROR (Context::addTube): Size of `nodes' and `radius' arguments must agree.");
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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addBox): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addBox): Texture file " + std::string(texturefile) + " does not exist.");
    }

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
    return addDisk(make_int2(Ndivs,1),center,size,make_SphericalCoord(0,0),make_RGBAcolor(1,0,0,1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation ){
    return addDisk(make_int2(Ndivs,1),center,size,rotation,make_RGBAcolor(1,0,0,1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color ){
    return addDisk(make_int2(Ndivs,1),center,size,rotation,make_RGBAcolor(color,1));
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color ){
    return addDisk(make_int2(Ndivs,1),center,size,rotation,color);
}

std::vector<uint> Context::addDisk(uint Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){
    return addDisk(make_int2(Ndivs,1),center,size,rotation,texture_file);
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBcolor &color ){
    return addDisk(Ndivs,center,size,rotation,make_RGBAcolor(color,1));
}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const RGBAcolor &color ){

    std::vector<uint> UUID;
    UUID.resize(Ndivs.x+Ndivs.x*(Ndivs.y-1)*2);
    int i=0;
    for( int r=0; r < Ndivs.y; r++ ) {
        for (int t = 0; t < Ndivs.x; t++) {

            float dtheta = 2.f * float(M_PI) / float(Ndivs.x);
            float theta = dtheta*float(t);
            float theta_plus = dtheta*float(t+1);

            float rx = size.x/float(Ndivs.y)*float(r);
            float ry = size.y/float(Ndivs.y)*float(r);

            float rx_plus = size.x/float(Ndivs.y)*float(r+1);
            float ry_plus = size.y/float(Ndivs.y)*float(r+1);

            if( r==0 ) {
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            }else{
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0),make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), color);
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), color);
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
    }

    return UUID;

}

std::vector<uint> Context::addDisk(const int2 &Ndivs, const vec3 &center, const vec2 &size, const SphericalCoord &rotation, const char* texture_file ){

    //texture must have type PNG or JPEG
    std::string fn = texture_file;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addDisk): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texture_file) ){
        helios_runtime_error("ERROR (Context::addDisk): Texture file " + std::string(texture_file) + " does not exist.");
    }

    std::vector<uint> UUID;
    UUID.resize(Ndivs.x+Ndivs.x*(Ndivs.y-1)*2);
    int i=0;
    for( int r=0; r < Ndivs.y; r++ ) {
        for (int t = 0; t < Ndivs.x; t++) {

            float dtheta = 2.f * float(M_PI) / float(Ndivs.x);
            float theta = dtheta*float(t);
            float theta_plus = dtheta*float(t+1);

            float rx = size.x/float(Ndivs.y)*float(r);
            float ry = size.y/float(Ndivs.y)*float(r);
            float rx_plus = size.x/float(Ndivs.y)*float(r+1);
            float ry_plus = size.y/float(Ndivs.y)*float(r+1);

            if( r==0 ) {
                UUID.at(i) = addTriangle(make_vec3(0, 0, 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texture_file, make_vec2(0.5,0.5), make_vec2(0.5f*(1.f+cosf(theta)*rx_plus/size.x), 0.5f*(1.f+sinf(theta)*ry_plus/size.y)), make_vec2(0.5f*(1.f+cosf(theta_plus)*rx_plus/size.x), 0.5f*(1.f+sinf(theta_plus)*ry_plus/size.y)));
            }else{
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx * cosf(theta), ry * sinf(theta), 0),make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0), texture_file, make_vec2(0.5f*(1.f+cosf(theta_plus)*rx/size.x), 0.5f*(1.f+sinf(theta_plus)*ry/size.y)), make_vec2(0.5f*(1.f+cosf(theta)*rx/size.x), 0.5f*(1.f+sinf(theta)*ry/size.y)), make_vec2(0.5f*(1.f+cosf(theta)*rx_plus/size.x), 0.5f*(1.f+sinf(theta)*ry_plus/size.y)));
                i++;
                UUID.at(i) = addTriangle(make_vec3(rx * cosf(theta_plus), ry * sinf(theta_plus), 0), make_vec3(rx_plus * cosf(theta), ry_plus * sinf(theta), 0),make_vec3(rx_plus * cosf(theta_plus), ry_plus * sinf(theta_plus), 0), texture_file, make_vec2(0.5f*(1.f+cosf(theta_plus)*rx/size.x), 0.5f*(1.f+sinf(theta_plus)*ry/size.y)), make_vec2(0.5f*(1.f+cosf(theta)*rx_plus/size.x), 0.5f*(1.f+sinf(theta)*ry_plus/size.y)), make_vec2(0.5f*(1.f+cosf(theta_plus)*rx_plus/size.x), 0.5f*(1.f+sinf(theta_plus)*ry_plus/size.y)));
            }
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.elevation, "y");
            getPrimitivePointer_private(UUID.at(i))->rotate(rotation.azimuth, "z");
            getPrimitivePointer_private(UUID.at(i))->translate(center);

            i++;
        }
    }

    return UUID;

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

    //texture must have type PNG or JPEG
    std::string fn = texturefile;
    std::string ext = getFileExtension(fn);
    if( ext != ".png" && ext != ".PNG" && ext != ".jpg" && ext != ".jpeg" && ext != ".JPG" && ext != ".JPEG" ){
        helios_runtime_error("ERROR (Context::addCone): Texture file " + fn + " is not PNG or JPEG format.");
    }else if( !doesTextureFileExist(texturefile) ){
        helios_runtime_error("ERROR (Context::addCone): Texture file " + std::string(texturefile) + " does not exist.");
    }

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

void Context::colorPrimitiveByDataPseudocolor( const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors ){

  colorPrimitiveByDataPseudocolor(UUIDs,primitive_data,colormap, Ncolors,9999999,-9999999);

}

void Context::colorPrimitiveByDataPseudocolor( const std::vector<uint> &UUIDs, const std::string &primitive_data, const std::string &colormap, uint Ncolors, float data_min, float data_max ){

  std::map<uint,float> pcolor_data;

  float data_min_new=9999999;
  float data_max_new=-9999999;
  for( uint UUID : UUIDs ){

    if( !doesPrimitiveExist(UUID) ){
      std::cout << "WARNING (Context::colorPrimitiveDataPseudocolor): primitive for UUID " << std::to_string(UUID) << " does not exist. Skipping this primitive." << std::endl;
      continue;
    }

    float dataf=0;
    if( doesPrimitiveDataExist(UUID,primitive_data.c_str()) ) {

      if( getPrimitiveDataType(UUID,primitive_data.c_str())!=HELIOS_TYPE_FLOAT && getPrimitiveDataType(UUID,primitive_data.c_str())!=HELIOS_TYPE_INT && getPrimitiveDataType(UUID,primitive_data.c_str())!=HELIOS_TYPE_UINT && getPrimitiveDataType(UUID,primitive_data.c_str())!=HELIOS_TYPE_DOUBLE  ){
        std::cout << "WARNING (Context::colorPrimitiveDataPseudocolor): Only primitive data types of int, uint, float, and double are supported for this function. Skipping this primitive." << std::endl;
        continue;
      }

      if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
          HELIOS_TYPE_FLOAT) {
        float data;
        getPrimitiveData(UUID, primitive_data.c_str(), data);
        dataf = data;
      } else if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                 HELIOS_TYPE_DOUBLE) {
        double data;
        getPrimitiveData(UUID, primitive_data.c_str(), data);
        dataf = float(data);
      } else if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                 HELIOS_TYPE_INT) {
        int data;
        getPrimitiveData(UUID, primitive_data.c_str(), data);
        dataf = float(data);
      } else if (getPrimitiveDataType(UUID, primitive_data.c_str()) ==
                 HELIOS_TYPE_UINT) {
        uint data;
        getPrimitiveData(UUID, primitive_data.c_str(), data);
        dataf = float(data);
      }
    }

    if( data_min==9999999 && data_max==-9999999 ) {
      if (dataf < data_min_new) {
        data_min_new = dataf;
      }
      if (dataf > data_max_new) {
        data_max_new = dataf;
      }
    }

    pcolor_data[UUID] = dataf;

  }

  if( data_min==9999999 && data_max==-9999999 ) {
    data_min = data_min_new;
    data_max = data_max_new;
  }

  std::vector<RGBcolor> colormap_data = generateColormap( colormap, Ncolors );

  std::map<std::string,std::vector<std::string> > cmap_texture_filenames;

  for( auto &data : pcolor_data ) {

    uint UUID = data.first;
    float pdata = data.second;

    std::string texturefile = getPrimitiveTextureFile(UUID);

    int cmap_ind = round((pdata - data_min) / (data_max - data_min) * float(Ncolors - 1));

    if( cmap_ind<0 ){
      cmap_ind=0;
    }else if( cmap_ind>=Ncolors ){
      cmap_ind=Ncolors-1;
    }

    if ( !texturefile.empty() && primitiveTextureHasTransparencyChannel(UUID)) { // primitive has texture with transparency channel

//      if (cmap_texture_filenames.find(texturefile) == cmap_texture_filenames.end()) {
//        cmap_texture_filenames[texturefile] = generateTexturesFromColormap(texturefile, colormap_data);
//      }
//
//      setPrimitiveTextureFile( UUID, cmap_texture_filenames.at(texturefile).at(cmap_ind));

      overridePrimitiveTextureColor(UUID);
      setPrimitiveColor(UUID, colormap_data.at(cmap_ind));

    } else { // primitive does not have texture with transparency channel - assign constant color

      if (!getPrimitiveTextureFile(UUID).empty()) {
        overridePrimitiveTextureColor(UUID);
      }

      setPrimitiveColor(UUID, colormap_data.at(cmap_ind));
    }
  }

}

std::vector<RGBcolor> Context::generateColormap( const std::vector<helios::RGBcolor> &ctable, const std::vector<float> &cfrac, uint Ncolors ){

  if( Ncolors>9999 ){
    std::cout << "WARNING (Context::generateColormapTextures): Truncating number of color map textures to maximum value of 9999." << std::endl;
  }

  if( ctable.size()!=cfrac.size() ){
    helios_runtime_error("ERROR (Context::generateColormap): The length of arguments 'ctable' and 'cfrac' must match.");
  }
  if( ctable.empty() ){
    helios_runtime_error("ERROR (Context::generateColormap): 'ctable' and 'cfrac' arguments contain empty vectors.");
  }

  std::vector<RGBcolor> color_table;
  color_table.resize(Ncolors);

  for (int i = 0; i < Ncolors; i++){

    float frac = float(i)/float(Ncolors-1)*cfrac.back();

    int j;
    for( j=0; j<cfrac.size()-1; j++ ){
      if( frac>=cfrac.at(j) && frac<=cfrac.at(j+1) ){
        break;
      }
    }

    float cminus = std::fmaxf(0.f,cfrac.at(j));
    float cplus = std::fminf(1.f,cfrac.at(j+1));

    float jfrac = (frac-cminus)/(cplus-cminus);

    RGBcolor color;
    color.r = ctable.at(j).r+jfrac*(ctable.at(j+1).r-ctable.at(j).r);
    color.g = ctable.at(j).g+jfrac*(ctable.at(j+1).g-ctable.at(j).g);
    color.b = ctable.at(j).b+jfrac*(ctable.at(j+1).b-ctable.at(j).b);

    color_table.at(i) = color;

  }

  return color_table;

}

std::vector<RGBcolor> Context::generateColormap( const std::string &colormap, uint Ncolors ){

  std::vector<RGBcolor> ctable_c;
  std::vector<float> clocs_c;

  if( colormap == "hot" ) {

    ctable_c.resize(5);
    ctable_c.at(0) = make_RGBcolor(0.f, 0.f, 0.f);
    ctable_c.at(1) = make_RGBcolor(0.5f, 0.f, 0.5f);
    ctable_c.at(2) = make_RGBcolor(1.f, 0.f, 0.f);
    ctable_c.at(3) = make_RGBcolor(1.f, 0.5f, 0.f);
    ctable_c.at(4) = make_RGBcolor(1.f, 1.f, 0.f);

    clocs_c.resize(5);
    clocs_c.at(0) = 0.f;
    clocs_c.at(1) = 0.25f;
    clocs_c.at(2) = 0.5f;
    clocs_c.at(3) = 0.75f;
    clocs_c.at(4) = 1.f;

  }else if( colormap == "cool") {

    ctable_c.resize(2);
    ctable_c.at(1) = RGB::cyan;
    ctable_c.at(2) = RGB::magenta;

    clocs_c.resize(2);
    clocs_c.at(1) = 0.f;
    clocs_c.at(2) = 1.f;

  }else if( colormap == "lava" ) {

    ctable_c.resize(5);
    ctable_c.at(0) = make_RGBcolor(0.f, 0.05f, 0.05f);
    ctable_c.at(1) = make_RGBcolor(0.f, 0.6f, 0.6f);
    ctable_c.at(2) = make_RGBcolor(1.f, 1.f, 1.f);
    ctable_c.at(3) = make_RGBcolor(1.f, 0.f, 0.f);
    ctable_c.at(4) = make_RGBcolor(0.5f, 0.f, 0.f);

    clocs_c.resize(5);
    clocs_c.at(0) = 0.f;
    clocs_c.at(1) = 0.4f;
    clocs_c.at(2) = 0.5f;
    clocs_c.at(3) = 0.6f;
    clocs_c.at(4) = 1.f;

  }else if( colormap == "rainbow" ){

    ctable_c.resize(4);
    ctable_c.at(0) = RGB::navy;
    ctable_c.at(1) = RGB::cyan;
    ctable_c.at(2) = RGB::yellow;
    ctable_c.at(3) = make_RGBcolor( 0.75f, 0.f, 0.f );

    clocs_c.resize(4);
    clocs_c.at(0) = 0.f;
    clocs_c.at(1) = 0.3f;
    clocs_c.at(2) = 0.7f;
    clocs_c.at(3) = 1.f;

  }else if( colormap == "parula" ){

    ctable_c.resize(4);
    ctable_c.at(0) = RGB::navy;
    ctable_c.at(1) = make_RGBcolor(0,0.6,0.6);
    ctable_c.at(2) = RGB::goldenrod;
    ctable_c.at(3) = RGB::yellow;

    clocs_c.resize(4);
    clocs_c.at(0) = 0.f;
    clocs_c.at(1) = 0.4f;
    clocs_c.at(2) = 0.7f;
    clocs_c.at(3) = 1.f;

  }else if( colormap == "gray" ) {

    ctable_c.resize(2);
    ctable_c.at(0) = RGB::black;
    ctable_c.at(1) = RGB::white;

    clocs_c.resize(2);
    clocs_c.at(0) = 0.f;
    clocs_c.at(1) = 1.f;

  }else if( colormap == "green" ) {

    ctable_c.resize(2);
    ctable_c.at(0) = RGB::black;
    ctable_c.at(1) = RGB::green;

    clocs_c.resize(2);
    clocs_c.at(0) = 0.f;
    clocs_c.at(1) = 1.f;

  }else{
    helios_runtime_error("ERROR (Context::generateColormapTextures): Unknown colormap "+colormap+".");
  }

  return generateColormap( ctable_c, clocs_c, Ncolors );

}

std::vector<std::string> Context::generateTexturesFromColormap( const std::string &texturefile, const std::vector<RGBcolor> &colormap_data ){

  uint Ncolors = colormap_data.size();

  // check that texture file exists
  std::ifstream tfile(texturefile);
  if (!tfile) {
    helios_runtime_error("ERROR (Context::generateTexturesFromColormap): Texture file " + texturefile + " does not exist, or you do not have permission to read it.");
  }
  tfile.close();

  // get file extension
  std::string file_ext = getFileExtension(texturefile);

  // get file base/stem
  std::string file_base = getFileStem(texturefile);

  std::vector<RGBcolor> color_table;
  color_table.resize(Ncolors);

  std::vector<std::string> texture_filenames(Ncolors);

  if (file_ext == "png" || file_ext == "PNG") {

    std::vector<RGBAcolor> pixel_data;
    uint width, height;
    readPNG(texturefile, width, height, pixel_data);

    for (int i = 0; i < Ncolors; i++) {

      std::ostringstream filename;
      filename << "lib/images/colormap_" << file_base << "_" << std::setw(4) << std::setfill('0') << std::to_string(i) << ".png";

      texture_filenames.at(i) = filename.str();

      RGBcolor color = colormap_data.at(i);

      for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
          pixel_data.at(row * width + col) = make_RGBAcolor(color, pixel_data.at(row * width + col).a);
        }
      }

      writePNG(filename.str(), width, height, pixel_data);
    }

  }

  return texture_filenames;

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
    
    uint current_objID = getPrimitivePointer_private(UUID)->getParentObjectID();
    getPrimitivePointer_private(UUID)->setParentObjectID(objID);
    
    if(current_objID != uint(0) && current_objID!=objID )
    {

        if( doesObjectExist(current_objID) ) {
            objects.at(current_objID)->deleteChildPrimitive(UUID);
            
            if(getObjectPointer_private(current_objID)->getPrimitiveUUIDs().size() == 0)
            {
                CompoundObject* obj = objects.at(current_objID);
                delete obj;
                objects.erase(current_objID);
                markGeometryDirty();
            }
        }
    }
    
}

void Context::setPrimitiveParentObjectID(const std::vector<uint> &UUIDs, uint objID) {
    for( uint UUID : UUIDs){
        setPrimitiveParentObjectID(UUID, objID);
    }
}

uint Context::getPrimitiveParentObjectID(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getParentObjectID();
}


std::vector<uint> Context::getUniquePrimitiveParentObjectIDs(std::vector<uint> UUIDs) const {
    return getUniquePrimitiveParentObjectIDs(UUIDs, false);
}


std::vector<uint> Context::getUniquePrimitiveParentObjectIDs(std::vector<uint> UUIDs, bool include_ObjID_zero) const {
    
    //vector of parent object ID for each primitive
    std::vector<uint> primitiveObjIDs;
    primitiveObjIDs.resize(UUIDs.size());
    for(uint i=0;i<UUIDs.size();i++)
    {
        primitiveObjIDs.at(i) = getPrimitivePointer_private(UUIDs.at(i))->getParentObjectID();
    }
    
    // sort
    std::sort(primitiveObjIDs.begin(), primitiveObjIDs.end());
    
    // unique
    auto it = unique(primitiveObjIDs.begin(), primitiveObjIDs.end());
    primitiveObjIDs.resize(distance(primitiveObjIDs.begin(), it));
    
    // remove object ID = 0 from the output if desired and it exisits
    if(include_ObjID_zero == false & primitiveObjIDs.at(0) == uint(0))
    {
        primitiveObjIDs.erase(primitiveObjIDs.begin()); 
    }
    
    return primitiveObjIDs;
}

float Context::getPrimitiveArea(uint UUID) const {
    return getPrimitivePointer_private(UUID)->getArea();
}

void Context::getPrimitiveBoundingBox( uint UUID, vec3 &min_corner, vec3 &max_corner ) const{
    std::vector<uint> UUIDs{UUID};
    getPrimitiveBoundingBox( UUIDs, min_corner, max_corner );
}

void Context::getPrimitiveBoundingBox( const std::vector<uint> &UUIDs, vec3 &min_corner, vec3 &max_corner ) const{


    uint p=0;
    for( uint UUID : UUIDs ){

        if ( !doesPrimitiveExist(UUID) ){
            helios_runtime_error("ERROR (Context::getPrimitiveBoundingBox): Primitive with UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }

        const std::vector<vec3> &vertices = getPrimitiveVertices(UUID);

        if( p==0 ){
            min_corner = vertices.front();
            max_corner = min_corner;
        }

        for ( const vec3 &vert : vertices ){
            if ( vert.x<min_corner.x ){
                min_corner.x = vert.x;
            }
            if ( vert.y<min_corner.y ){
                min_corner.y = vert.y;
            }
            if ( vert.z<min_corner.z ){
                min_corner.z = vert.z;
            }
            if ( vert.x>max_corner.x ){
                max_corner.x = vert.x;
            }
            if ( vert.y>max_corner.y ){
                max_corner.y = vert.y;
            }
            if ( vert.z>max_corner.z ){
                max_corner.z = vert.z;
            }
        }

        p++;
    }
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

void Context::setPrimitiveTextureFile( uint UUID, const std::string &texturefile ){
  getPrimitivePointer_private(UUID)->setTextureFile(texturefile.c_str());
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
        helios_runtime_error("ERROR (getPrimitiveTransparencyData): Texture transparency data does not exist for primitive " + std::to_string(UUID) + ".");
        return 0;
    }
}

void Context::overridePrimitiveTextureColor( uint UUID ){
    getPrimitivePointer_private(UUID)->overrideTextureColor();
}

void Context::overridePrimitiveTextureColor( const std::vector<uint> &UUIDs ){
    for( uint UUID : UUIDs ) {
        getPrimitivePointer_private(UUID)->overrideTextureColor();
    }
}

void Context::usePrimitiveTextureColor( uint UUID ){
    getPrimitivePointer_private(UUID)->useTextureColor();
}

void Context::usePrimitiveTextureColor( const std::vector<uint> &UUIDs ){
    for( uint UUID : UUIDs ) {
        getPrimitivePointer_private(UUID)->useTextureColor();
    }
}

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
    std::cout << "Parent ObjID: " << getPrimitiveParentObjectID(UUID) << std::endl;
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
    // Primitive* pointer = getPrimitivePointer_private(UUID);
    std::vector<std::string> pd = listPrimitiveData(UUID);
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
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
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else{
            assert(false);
        }
        
    }
    std::cout << "-------------------------------------------" << std::endl;
}

void Context::printObjectInfo(uint ObjID) const{
    
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "Info for ObjID " << ObjID << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    
    ObjectType otype = getObjectType(ObjID);
    std::string ostype;
    if( otype == 0){
        ostype =  "OBJECT_TYPE_TILE";
    }else if(otype == 1){
        ostype =  "OBJECT_TYPE_SPHERE";
    }else if(otype == 2){
        ostype =  "OBJECT_TYPE_TUBE";
    }else if(otype == 3){
        ostype =  "OBJECT_TYPE_BOX";
    }else if(otype == 4){
        ostype =  "OBJECT_TYPE_DISK";
    }else if(otype == 5){
        ostype =  "OBJECT_TYPE_POLYMESH";
    }else if(otype == 6){
        ostype =  "OBJECT_TYPE_CONE";
    }
    
    std::cout << "Type: " << ostype << std::endl;
    std::cout << "Object Bounding Box Center: " << getObjectCenter(ObjID) << std::endl;
    std::cout << "One-sided Surface Area: " << getObjectArea(ObjID) << std::endl;
    
    std::cout << "Primitive Count: " << getObjectPrimitiveCount(ObjID) << std::endl;
    
    if(areObjectPrimitivesComplete(ObjID))
    {
        std::cout << "Object Primitives Complete" << std::endl; 
    }else{
        std::cout << "Object Primitives Incomplete" << std::endl;
    }
    
    std::cout << "Primitive UUIDs: " << std::endl;
    std::vector<uint> primitive_UUIDs = getObjectPrimitiveUUIDs(ObjID);
    for(uint i=0; i<primitive_UUIDs.size();i++)
    {
        if(i < 5){
            PrimitiveType ptype = getPrimitiveType(primitive_UUIDs.at(i));
            std::string pstype;
            if( ptype == 0){
                pstype =  "PRIMITIVE_TYPE_PATCH";
            }else if(ptype == 1){
                pstype =  "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(i) << " (" << pstype << ")" << std::endl;
        }else{
            std::cout << "   ..." << std::endl;
            PrimitiveType ptype = getPrimitiveType(primitive_UUIDs.at(primitive_UUIDs.size()-2));
            std::string pstype;
            if( ptype == 0){
                pstype =  "PRIMITIVE_TYPE_PATCH";
            }else if(ptype == 1){
                pstype =  "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(primitive_UUIDs.size()-2) << " (" << pstype << ")" << std::endl;
            ptype = getPrimitiveType(primitive_UUIDs.at(primitive_UUIDs.size()-1));
            if( ptype == 0){
                pstype =  "PRIMITIVE_TYPE_PATCH";
            }else if(ptype == 1){
                pstype =  "PRIMITIVE_TYPE_TRIANGLE";
            }
            std::cout << "   " << primitive_UUIDs.at(primitive_UUIDs.size()-1) << " (" << pstype << ")" << std::endl;
            break;
        }
    }
    
    if(otype == OBJECT_TYPE_TILE)
    {
        std::cout << "Tile Center: " << getTileObjectCenter(ObjID) << std::endl;
        std::cout << "Tile Size: " << getTileObjectSize(ObjID) << std::endl;
        std::cout << "Tile Subdivision Count: " << getTileObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Tile Normal: " << getTileObjectNormal(ObjID) << std::endl;
        
        std::cout << "Tile Texture UV: " << std::endl;
        std::vector<vec2> uv = getTileObjectTextureUV(ObjID);
        for(uint i=0; i<uv.size();i++)
        {
            std::cout << "   " << uv.at(i) << std::endl;
        }
        
        std::cout << "Tile Vertices: " << std::endl;
        std::vector<vec3> primitive_vertices = getTileObjectVertices(ObjID);
        for(uint i=0; i<primitive_vertices.size();i++)
        {
            std::cout << "   " << primitive_vertices.at(i) << std::endl;
        }
        
        
    }else if(otype == OBJECT_TYPE_SPHERE){
        
        std::cout << "Sphere Center: " << getSphereObjectCenter(ObjID) << std::endl;
        std::cout << "Sphere Radius: " << getSphereObjectRadius(ObjID) << std::endl;
        std::cout << "Sphere Subdivision Count: " << getSphereObjectSubdivisionCount(ObjID) << std::endl;
        
    }else if(otype == OBJECT_TYPE_TUBE){
        
        std::cout << "Tube Subdivision Count: " << getTubeObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Tube Nodes: " << std::endl;
        std::vector<vec3> nodes = getTubeObjectNodes(ObjID);
        for(uint i=0; i<nodes.size();i++)
        {
            if(i < 10){
                std::cout << "   " << nodes.at(i) << std::endl;
            }else{
                std::cout << "      ..." << std::endl;
                std::cout << "   " << nodes.at(nodes.size()-2) << std::endl;
                std::cout << "   " << nodes.at(nodes.size()-1) << std::endl;
                break;
            }
        }
        std::cout << "Tube Node Radii: " << std::endl;
        std::vector<float> noderadii = getTubeObjectNodeRadii(ObjID);
        for(uint i=0; i<noderadii.size();i++)
        {
            if(i < 10){
                std::cout << "   " << noderadii.at(i) << std::endl;
            }else{
                std::cout << "      ..." << std::endl;
                std::cout << "   " << noderadii.at(noderadii.size()-2) << std::endl;
                std::cout << "   " << noderadii.at(noderadii.size()-1) << std::endl;
                break;
            }
        }
        std::cout << "Tube Node Colors: " << std::endl;
        std::vector<helios::RGBcolor> nodecolors = getTubeObjectNodeColors(ObjID);
        for(uint i=0; i<nodecolors.size();i++)
        {
            if(i < 10){
                std::cout << "   " << nodecolors.at(i) << std::endl;
            }else{
                std::cout << "      ..." << std::endl;
                std::cout << "      " << nodecolors.at(nodecolors.size()-2) << std::endl;
                std::cout << "      " << nodecolors.at(nodecolors.size()-1) << std::endl;
                break;
            }
        }
        
    }else if(otype == OBJECT_TYPE_BOX){
        
        std::cout << "Box Center: " << getBoxObjectCenter(ObjID) << std::endl;
        std::cout << "Box Size: " << getBoxObjectSize(ObjID) << std::endl;
        std::cout << "Box Subdivision Count: " << getBoxObjectSubdivisionCount(ObjID) << std::endl;
        
    }else if(otype == OBJECT_TYPE_DISK){
        
        std::cout << "Disk Center: " << getDiskObjectCenter(ObjID) << std::endl;
        std::cout << "Disk Size: " << getDiskObjectSize(ObjID) << std::endl;
        std::cout << "Disk Subdivision Count: " << getDiskObjectSubdivisionCount(ObjID) << std::endl;
        
        // }else if(type == OBJECT_TYPE_POLYMESH){
        // nothing for now
        
    }else if(otype == OBJECT_TYPE_CONE){
        
        std::cout << "Cone Length: " << getConeObjectLength(ObjID) << std::endl;
        std::cout << "Cone Axis Unit Vector: " << getConeObjectAxisUnitVector(ObjID) << std::endl;
        std::cout << "Cone Subdivision Count: " << getConeObjectSubdivisionCount(ObjID) << std::endl;
        std::cout << "Cone Nodes: " << std::endl;
        std::vector<vec3> nodes = getConeObjectNodes(ObjID);
        for(uint i=0; i<nodes.size();i++)
        {
            std::cout << "   " << nodes.at(i) << std::endl;
        }
        std::cout << "Cone Node Radii: " << std::endl;
        std::vector<float> noderadii = getConeObjectNodeRadii(ObjID);
        for(uint i=0; i<noderadii.size();i++)
        {
            std::cout << "   " << noderadii.at(i) << std::endl;
        }
    }
    
    
    float T[16];
    getObjectTransformationMatrix(ObjID, T);
    std::cout << "Transform: " << std::endl;
    std::cout << "   " << T[0] << "      " << T[1] << "      " << T[2] << "      " << T[3] << std::endl;
    std::cout << "   " << T[4] << "      " << T[5] << "      " << T[6] << "      " << T[7] << std::endl;
    std::cout << "   " << T[8] << "      " << T[9] << "      " << T[10] << "      " << T[11] << std::endl;
    std::cout << "   " << T[12] << "      " << T[13] << "      " << T[14] << "      " << T[15] << std::endl;

    std::cout << "Texture File: " << getObjectTextureFile(ObjID) << std::endl;
    
    std::cout << "Object Data: " << std::endl;
    // Primitive* pointer = getPrimitivePointer_private(ObjID);
    std::vector<std::string> pd = listObjectData(ObjID);
    for(uint i=0; i<pd.size();i++)
    {
        uint dsize = getObjectDataSize(ObjID, pd.at(i).c_str());
        HeliosDataType dtype = getObjectDataType(ObjID, pd.at(i).c_str());
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
            getObjectData( ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_UINT ){
            std::vector<uint> pdata;
            getObjectData( ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_FLOAT ){
            std::vector<float> pdata;
            getObjectData( ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_DOUBLE ){
            std::vector<double> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_VEC2 ){
            std::vector<vec2> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_VEC3 ){
            std::vector<vec3> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_VEC4 ){
            std::vector<vec4> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_INT2 ){
            std::vector<int2> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_INT3 ){
            std::vector<int3> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
            for(uint j=0; j<dsize;j++)
            {
                if(j < 10){
                    std::cout << "      " << pdata.at(j) << std::endl;
                }else{
                    std::cout << "      ..." << std::endl;
                    std::cout << "      " << pdata.at(dsize-2) << std::endl;
                    std::cout << "      " << pdata.at(dsize-1) << std::endl;
                    break;
                }
                
            }
        }else if( dtype==HELIOS_TYPE_INT4 ){
            std::vector<int4> pdata;
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
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
            getObjectData(ObjID, pd.at(i).c_str(), pdata );
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

CompoundObject* Context::getObjectPointer_private( uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }
    return objects.at(ObjID);
}


float Context::getObjectArea(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getArea();
}

uint Context::getObjectPrimitiveCount(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getPrimitiveCount();
}

helios::vec3 Context::getObjectCenter(uint ObjID) const {
    return getObjectPointer_private(ObjID)->getObjectCenter();
}

std::string Context::getObjectTextureFile(uint ObjID) const{
    return getObjectPointer_private(ObjID)->getTextureFile();
}

void Context::getObjectTransformationMatrix(uint ObjID, float (&T)[16] ) const {
    getObjectPointer_private(ObjID)->getTransformationMatrix( T );
}

void Context::setObjectTransformationMatrix(uint ObjID, float (&T)[16] ) {
    getObjectPointer_private(ObjID)->setTransformationMatrix(T);
}

void Context::setObjectTransformationMatrix(const std::vector<uint> &ObjIDs, float (&T)[16] ) {
    for( uint ObjID : ObjIDs){
        getObjectPointer_private(ObjID)->setTransformationMatrix(T);
    }
}

bool Context::objectHasTexture( uint ObjID ) const{
    return getObjectPointer_private(ObjID)->hasTexture();
}

void Context::setObjectColor(uint ObjID, const RGBcolor &color) {
    getObjectPointer_private(ObjID)->setColor( color );
}

void Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBcolor &color) {
    for( uint ObjID : ObjIDs){
        getObjectPointer_private(ObjID)->setColor(color);
    }
}

void Context::setObjectColor(uint ObjID, const RGBAcolor &color) {
    getObjectPointer_private(ObjID)->setColor( color );
}

void Context::setObjectColor(const std::vector<uint> &ObjIDs, const RGBAcolor &color) {
    for( uint ObjID : ObjIDs){
        getObjectPointer_private(ObjID)->setColor(color);
    }
}

bool Context::doesObjectContainPrimitive(uint ObjID, uint UUID ){
    return getObjectPointer_private(ObjID)->doesObjectContainPrimitive( UUID );
}

void Context::overrideObjectTextureColor(uint ObjID) {
    getObjectPointer_private(ObjID)->overrideTextureColor();
}

void Context::overrideObjectTextureColor( const std::vector<uint> &ObjIDs) {
    for( uint ObjID : ObjIDs ) {
        getObjectPointer_private(ObjID)->overrideTextureColor();
    }
}

void Context::useObjectTextureColor(uint ObjID) {
    getObjectPointer_private(ObjID)->useTextureColor();
}

void Context::useObjectTextureColor( const std::vector<uint> &ObjIDs) {
    for( uint ObjID : ObjIDs ) {
        getObjectPointer_private(ObjID)->useTextureColor();
    }
}

void Context::getObjectBoundingBox( uint ObjID, vec3 &min_corner, vec3 &max_corner ) const {
    std::vector<uint> ObjIDs{ObjID};
    getObjectBoundingBox(ObjIDs,min_corner,max_corner);
}

void Context::getObjectBoundingBox( const std::vector<uint> &ObjIDs, vec3 &min_corner, vec3 &max_corner ) const{

    uint o=0;
    for( uint ObjID : ObjIDs ) {

        if ( objects.find(ObjID) == objects.end()){
            helios_runtime_error("ERROR (getObjectBoundingBox): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
        }

        const std::vector<uint> &UUIDs = objects.at(ObjID)->getPrimitiveUUIDs();

        uint p=0;
        for (uint UUID: UUIDs) {

            const std::vector<vec3> &vertices = getPrimitiveVertices(UUID);

            if( p==0 && o==0 ){
                min_corner = vertices.front();
                max_corner = min_corner;
                p++;
                continue;
            }

            for (const vec3 &vert: vertices) {
                if (vert.x < min_corner.x) {
                    min_corner.x = vert.x;
                }
                if (vert.y < min_corner.y) {
                    min_corner.y = vert.y;
                }
                if (vert.z < min_corner.z) {
                    min_corner.z = vert.z;
                }
                if (vert.x > max_corner.x) {
                    max_corner.x = vert.x;
                }
                if (vert.y > max_corner.y) {
                    max_corner.y = vert.y;
                }
                if (vert.z > max_corner.z) {
                    max_corner.z = vert.z;
                }
            }
        }

        o++;
    }

}

Tile* Context::getTileObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_TILE ){
        helios_runtime_error("ERROR (getTileObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Tile Object.");
    }
    return dynamic_cast<Tile*>(objects.at(ObjID));
}

Sphere* Context::getSphereObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_SPHERE ){
        helios_runtime_error("ERROR (getSphereObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Sphere Object.");
    }
    return dynamic_cast<Sphere*>(objects.at(ObjID));
}

Tube* Context::getTubeObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_TUBE ){
        helios_runtime_error("ERROR (getTubeObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Tube Object.");
    }
    return dynamic_cast<Tube*>(objects.at(ObjID));
}

Box* Context::getBoxObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_BOX ){
        helios_runtime_error("ERROR (getBoxObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Box Object.");
    }
    return dynamic_cast<Box*>(objects.at(ObjID));
}

Disk* Context::getDiskObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_DISK ){
        helios_runtime_error("ERROR (getDiskObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Disk Object.");
    }
    return dynamic_cast<Disk*>(objects.at(ObjID));
}

Polymesh* Context::getPolymeshObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_POLYMESH ){
        helios_runtime_error("ERROR (getPolymeshObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Polymesh Object.");
    }
    return dynamic_cast<Polymesh*>(objects.at(ObjID));
}

Cone* Context::getConeObjectPointer_private(uint ObjID ) const{
    if( objects.find(ObjID) == objects.end() ){
        helios_runtime_error("ERROR (getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " does not exist in the Context.");
    }else if( objects.at(ObjID)->getObjectType()!=OBJECT_TYPE_CONE ){
        helios_runtime_error("ERROR (getConeObjectPointer): ObjectID of " + std::to_string(ObjID) + " is not a Cone Object.");
    }
    return dynamic_cast<Cone*>(objects.at(ObjID));
}

helios::vec3 Context::getTileObjectCenter(uint &ObjID) const {
    return getTileObjectPointer_private(ObjID)->getCenter();
}

helios::vec2 Context::getTileObjectSize(uint &ObjID) const {
    return getTileObjectPointer_private(ObjID)->getSize();
}

helios::int2 Context::getTileObjectSubdivisionCount(uint &ObjID) const {
    return getTileObjectPointer_private(ObjID)->getSubdivisionCount();
}

helios::vec3 Context::getTileObjectNormal(uint &ObjID) const {
    return getTileObjectPointer_private(ObjID)->getNormal();
}

std::vector<helios::vec2> Context::getTileObjectTextureUV(uint &ObjID) const {
    return getTileObjectPointer_private(ObjID)->getTextureUV();
}

std::vector<helios::vec3> Context::getTileObjectVertices(uint &ObjID) const {
    return getTileObjectPointer_private(ObjID)->getVertices();
}

helios::vec3 Context::getSphereObjectCenter(uint &ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getCenter();
}

helios::vec3 Context::getSphereObjectRadius(uint &ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getRadius();
}

uint Context::getSphereObjectSubdivisionCount(uint &ObjID) const {
    return getSphereObjectPointer_private(ObjID)->getSubdivisionCount();
}

uint Context::getTubeObjectSubdivisionCount(uint &ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getSubdivisionCount();
}

std::vector<helios::vec3> Context::getTubeObjectNodes(uint &ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodes();
}

std::vector<float> Context::getTubeObjectNodeRadii(uint &ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeRadii();
}

std::vector<RGBcolor> Context::getTubeObjectNodeColors(uint &ObjID) const {
    return getTubeObjectPointer_private(ObjID)->getNodeColors();
}

helios::vec3 Context::getBoxObjectCenter(uint &ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getCenter();
}

helios::vec3 Context::getBoxObjectSize(uint &ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getSize();
}

helios::int3 Context::getBoxObjectSubdivisionCount(uint &ObjID) const {
    return getBoxObjectPointer_private(ObjID)->getSubdivisionCount();
}

helios::vec3 Context::getDiskObjectCenter(uint &ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getCenter();
}

helios::vec2 Context::getDiskObjectSize(uint &ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getSize();
}

uint Context::getDiskObjectSubdivisionCount(uint &ObjID) const {
    return getDiskObjectPointer_private(ObjID)->getSubdivisionCount().x;
}

uint Context::getConeObjectSubdivisionCount(uint &ObjID) const {
    return getConeObjectPointer_private(ObjID)->getSubdivisionCount();
}

std::vector<helios::vec3> Context::getConeObjectNodes(uint &ObjID) const {
    return getConeObjectPointer_private(ObjID)->getNodes();
}

std::vector<float> Context::getConeObjectNodeRadii(uint &ObjID) const {
    return getConeObjectPointer_private(ObjID)->getNodeRadii();
}

helios::vec3 Context::getConeObjectNode(uint &ObjID, int number) const {
    return getConeObjectPointer_private(ObjID)->getNode(number);
}

float Context::getConeObjectNodeRadius(uint &ObjID, int number) const {
    return getConeObjectPointer_private(ObjID)->getNodeRadius(number);
}

helios::vec3 Context::getConeObjectAxisUnitVector(uint &ObjID) const {
    return getConeObjectPointer_private(ObjID)->getAxisUnitVector();
}

float Context::getConeObjectLength(uint &ObjID) const {
    return getConeObjectPointer_private(ObjID)->getLength();
}

void Context::duplicatePrimitiveData( const char* existing_data_label, const char* copy_data_label ){

    for( auto primitive : primitives){
        if( primitive.second->doesPrimitiveDataExist(existing_data_label) ){
            HeliosDataType type = primitive.second->getPrimitiveDataType(existing_data_label);
            if( type==HELIOS_TYPE_FLOAT ){
                std::vector<float> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_DOUBLE ) {
                std::vector<double> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_INT ) {
                std::vector<int> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_UINT ) {
                std::vector<uint> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_VEC2 ) {
                std::vector<vec2> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_VEC3 ) {
                std::vector<vec3> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_VEC4 ) {
                std::vector<vec4> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_INT2 ) {
                std::vector<int2> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_INT3 ) {
                std::vector<int3> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }else if( type==HELIOS_TYPE_STRING ){
                std::vector<std::string> data;
                primitive.second->getPrimitiveData(existing_data_label, data);
                primitive.second->setPrimitiveData(copy_data_label, type, data.size(), &data.front());
            }
        }
    }


}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, float &mean ) const{
  float value;
  float sum = 0.f;
  size_t count = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum += value;
      count++;
    }

  }

  if( count==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    mean = 0;
  }else{
    mean = sum/float(count);
  }

}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, double &mean ) const{
  double value;
  double sum = 0.f;
  size_t count = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum += value;
      count++;
    }

  }

  if( count==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    mean = 0;
  }else{
    mean = sum/float(count);
  }

}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &mean ) const{
  vec2 value;
  vec2 sum(0.f,0.f);
  size_t count = 0;
  for (uint UUID : UUIDs) {

    if ( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(UUID, label.c_str()) == HELIOS_TYPE_VEC2 ) {
      getPrimitiveData(UUID, label.c_str(), value);
      sum = sum + value;
      count++;
    }
  }

  if (count == 0) {
    std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    mean = make_vec2(0,0);
  } else {
    mean = sum / float(count);
  }
}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &mean ) const{
  vec3 value;
  vec3 sum(0.f,0.f,0.f);
  size_t count = 0;
  for (uint UUID : UUIDs) {

    if ( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(UUID, label.c_str()) == HELIOS_TYPE_VEC3 ) {
      getPrimitiveData(UUID, label.c_str(), value);
      sum = sum + value;
      count++;
    }
  }

  if (count == 0) {
    std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    mean = make_vec3(0,0,0);
  } else {
    mean = sum / float(count);
  }
}

void Context::calculatePrimitiveDataMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &mean ) const{
  vec4 value;
  vec4 sum(0.f,0.f,0.f,0.f);
  size_t count = 0;
  for (uint UUID : UUIDs) {

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(UUID, label.c_str()) == HELIOS_TYPE_VEC4 ) {
      getPrimitiveData(UUID, label.c_str(), value);
      sum = sum + value;
      count++;
    }
  }

  if (count == 0) {
    std::cout << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    mean = make_vec4(0,0,0,0);
  } else {
    mean = sum / float(count);
  }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, float &awt_mean ) const{
  float value, A;
  float sum = 0.f;
  float area = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
      getPrimitiveData(UUID,label.c_str(),value);
      A = getPrimitiveArea(UUID);
      sum += value*A;
      area += A;
    }

  }

  if( area==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    awt_mean = 0;
  }else{
    awt_mean = sum/area;
  }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, double &awt_mean ) const{
  double value;
  float A;
  double sum = 0.f;
  double area = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
      getPrimitiveData(UUID,label.c_str(),value);
      A = getPrimitiveArea(UUID);
      sum += value*double(A);
      area += A;
    }

  }

  if( area==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    awt_mean = 0;
  }else{
    awt_mean = sum/area;
  }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_mean ) const{
  vec2 value;
  float A;
  vec2 sum(0.f,0.f);
  float area = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC2 ){
      getPrimitiveData(UUID,label.c_str(),value);
      A = getPrimitiveArea(UUID);
      sum = sum + (value*A);
      area += A;
    }

  }

  if( area==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    awt_mean = make_vec2(0,0);
  }else{
    awt_mean = sum/area;
  }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_mean ) const{
  vec3 value;
  float A;
  vec3 sum(0.f,0.f,0.f);
  float area = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC3 ){
      getPrimitiveData(UUID,label.c_str(),value);
      A = getPrimitiveArea(UUID);
      sum = sum + (value*A);
      area += A;
    }

  }

  if( area==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    awt_mean = make_vec3(0,0,0);
  }else{
    awt_mean = sum/area;
  }
}

void Context::calculatePrimitiveDataAreaWeightedMean( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_mean ) const{
  vec4 value;
  float A;
  vec4 sum(0.f,0.f,0.f,0.f);
  float area = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC4 ){
      getPrimitiveData(UUID,label.c_str(),value);
      A = getPrimitiveArea(UUID);
      sum = sum + (value*A);
      area += A;
    }

  }

  if( area==0 ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    awt_mean = make_vec4(0,0,0,0);
  }else{
    awt_mean = sum/area;
  }
}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, float &sum ) const{

  float value;
  sum = 0.f;
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum += value;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, double &sum ) const{

  double value;
  sum = 0.f;
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum += value;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &sum ) const{

  vec2 value;
  sum = make_vec2(0.f,0.f);
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC2 ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum = sum + value;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &sum ) const{

  vec3 value;
  sum = make_vec3(0.f,0.f,0.f);
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC3 ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum = sum + value;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &sum ) const{

  vec4 value;
  sum = make_vec4(0.f,0.f,0.f,0.f);
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID)  && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC4 ){
      getPrimitiveData(UUID,label.c_str(),value);
      sum = sum + value;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, float &awt_sum ) const{

  float value;
  awt_sum = 0.f;
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_FLOAT ){
      float area = getPrimitiveArea(UUID);
      getPrimitiveData(UUID,label.c_str(),value);
      awt_sum += value*area;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, double &awt_sum ) const{

  double value;
  awt_sum = 0.f;
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_DOUBLE ){
      float area = getPrimitiveArea(UUID);
      getPrimitiveData(UUID,label.c_str(),value);
      awt_sum += value*area;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_sum ) const{

  vec2 value;
  awt_sum = make_vec2(0.f,0.f);
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC2 ){
      float area = getPrimitiveArea(UUID);
      getPrimitiveData(UUID,label.c_str(),value);
      awt_sum = awt_sum + value*area;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_sum ) const{

  vec3 value;
  awt_sum = make_vec3(0.f,0.f,0.f);
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC3 ){
      float area = getPrimitiveArea(UUID);
      getPrimitiveData(UUID,label.c_str(),value);
      awt_sum = awt_sum + value*area;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::calculatePrimitiveDataAreaWeightedSum( const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_sum ) const{

  vec4 value;
  awt_sum = make_vec4(0.f,0.f,0.f,0.F);
  bool added_to_sum = false;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID)  && doesPrimitiveDataExist(UUID,label.c_str()) && getPrimitiveDataType(UUID,label.c_str())==HELIOS_TYPE_VEC4 ){
      float area = getPrimitiveArea(UUID);
      getPrimitiveData(UUID,label.c_str(),value);
      awt_sum = awt_sum + value*area;
      added_to_sum = true;
    }

  }

  if( !added_to_sum ) {
    std::cout << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
  }

}

void Context::scalePrimitiveData( const std::vector<uint> &UUIDs, const std::string &label, float scaling_factor ){

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;
    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            primitives_not_exist++;
            continue;
        }
        if( !doesPrimitiveDataExist(UUID, label.c_str()) ){
            primitive_data_not_exist++;
            continue;
        }
        HeliosDataType data_type = getPrimitiveDataType(UUID,label.c_str());
        if( data_type==HELIOS_TYPE_FLOAT ){
            float data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_DOUBLE ){
            double data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_VEC2 ){
            vec2 data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_VEC3 ){
            vec3 data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else if( data_type==HELIOS_TYPE_VEC4 ){
            vec4 data;
            primitives.at(UUID)->getPrimitiveData(label.c_str(),data);
            primitives.at(UUID)->setPrimitiveData(label.c_str(), data*scaling_factor );
        }else{
            helios_runtime_error("ERROR (Context::scalePrimitiveData): This operation only supports primitive data of type float, double, vec2, vec3, and vec4.");
        }
    }

    if( primitives_not_exist>0 ){
        std::cout << "WARNING (Context::scalePrimitiveData): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if( primitive_data_not_exist>0 ){
        std::cout << "WARNING (Context::scalePrimitiveData): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no scaling was applied." << std::endl;
    }

}

void Context::aggregatePrimitiveDataSum( const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label  ){

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;

    float data_float = 0;
    double data_double = 0;
    uint data_uint = 0;
    int data_int = 0;
    int2 data_int2;
    int3 data_int3;
    int4 data_int4;
    vec2 data_vec2;
    vec3 data_vec3;
    vec4 data_vec4;

    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            primitives_not_exist++;
            continue;
        }

        HeliosDataType data_type;

        bool init_type = false;
        for( const auto &label : primitive_data_labels ) {

            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                continue;
            }

            HeliosDataType data_type_current = getPrimitiveDataType(UUID, label.c_str());
            if( !init_type ) {
                data_type = data_type_current;
                init_type = true;
            }else{
                if( data_type!=data_type_current ){
                    helios_runtime_error("ERROR (Context::aggregatePrimitiveDataSum): Primitive data types are not consistent for UUID " + std::to_string(UUID));
                }
            }

            if ( data_type_current == HELIOS_TYPE_FLOAT) {
                float data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_float += data;
            } else if ( data_type_current == HELIOS_TYPE_DOUBLE) {
                double data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_double += data;
            } else if ( data_type_current == HELIOS_TYPE_VEC2) {
                vec2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec2 = data_vec2 + data;
            } else if ( data_type_current == HELIOS_TYPE_VEC3) {
                vec3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec3 = data_vec3 + data;
            } else if ( data_type_current == HELIOS_TYPE_VEC4) {
                vec4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec4 = data_vec4 + data;
            } else if ( data_type_current == HELIOS_TYPE_INT) {
                int data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int = data_int + data;
            } else if ( data_type_current == HELIOS_TYPE_UINT) {
                uint data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_uint = data_uint + data;
            } else if ( data_type_current == HELIOS_TYPE_INT2) {
                int2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int2 = data_int2 + data;
            } else if ( data_type_current == HELIOS_TYPE_INT3) {
                int3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int3 = data_int3 + data;
            } else if ( data_type_current == HELIOS_TYPE_INT4) {
                int4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int4 = data_int4 + data;
            } else {
                helios_runtime_error("ERROR (Context::aggregatePrimitiveDataSum): This operation is not supported for string primitive data types.");
            }
        }

        if( !init_type ){
            primitive_data_not_exist++;
            continue;
        }else if ( data_type == HELIOS_TYPE_FLOAT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_float );
            data_float = 0;
        } else if ( data_type == HELIOS_TYPE_DOUBLE) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_double );
            data_double = 0;
        } else if ( data_type == HELIOS_TYPE_VEC2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec2 );
            data_vec2 = make_vec2(0,0);
        } else if ( data_type == HELIOS_TYPE_VEC3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec3 );
            data_vec3 = make_vec3(0,0,0);
        } else if ( data_type == HELIOS_TYPE_VEC4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec4 );
            data_vec4 = make_vec4(0,0,0,0);
        } else if ( data_type == HELIOS_TYPE_INT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int );
            data_int = 0;
        } else if ( data_type == HELIOS_TYPE_UINT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_uint );
            data_uint = 0;
        } else if ( data_type == HELIOS_TYPE_INT2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int2 );
            data_int2 = make_int2(0,0);
        } else if ( data_type == HELIOS_TYPE_INT3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int3 );
            data_int3 = make_int3(0,0,0);
        } else if ( data_type == HELIOS_TYPE_INT4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int4 );
            data_int4 = make_int4(0,0,0,0);
        }

    }

    if( primitives_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataSum): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if( primitive_data_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataSum): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no scaling summation was performed and new primitive data was not created for this primitive." << std::endl;
    }

}

void Context::aggregatePrimitiveDataProduct( const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label  ){

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;

    float data_float = 0;
    double data_double = 0;
    uint data_uint = 0;
    int data_int = 0;
    int2 data_int2;
    int3 data_int3;
    int4 data_int4;
    vec2 data_vec2;
    vec3 data_vec3;
    vec4 data_vec4;

    for( uint UUID : UUIDs ){
        if( !doesPrimitiveExist(UUID) ){
            primitives_not_exist++;
            continue;
        }

        HeliosDataType data_type;

        bool init_type = false;
        int i=0;
        for( const auto &label : primitive_data_labels ) {

            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                continue;
            }

            HeliosDataType data_type_current = getPrimitiveDataType(UUID, label.c_str());
            if( !init_type ) {
                data_type = data_type_current;
                init_type = true;
            }else{
                if( data_type!=data_type_current ){
                    helios_runtime_error("ERROR (Context::aggregatePrimitiveDataProduct): Primitive data types are not consistent for UUID " + std::to_string(UUID));
                }
            }

            if ( data_type_current == HELIOS_TYPE_FLOAT) {
                float data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_float = data;
                }else {
                    data_float *= data;
                }
            } else if ( data_type_current == HELIOS_TYPE_DOUBLE) {
                double data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ) {
                    data_double *= data;
                }else{
                    data_double = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_VEC2) {
                vec2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_vec2.x *= data.x;
                    data_vec2.y *= data.y;
                }else{
                    data_vec2 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_VEC3) {
                vec3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_vec3.x *= data.x;
                    data_vec3.y *= data.y;
                    data_vec3.z *= data.z;
                }else{
                    data_vec3 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_VEC4) {
                vec4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_vec4.x *= data.x;
                    data_vec4.y *= data.y;
                    data_vec4.z *= data.z;
                    data_vec4.w *= data.w;
                }else{
                    data_vec4 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT) {
                int data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int = data_int * data;
                }else{
                    data_int = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_UINT) {
                uint data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_uint = data_uint * data;
                }else{
                    data_uint = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT2) {
                int2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int2.x *= data.x;
                    data_int2.y *= data.y;
                }else{
                    data_int2 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT3) {
                int3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int3.x *= data.x;
                    data_int3.y *= data.y;
                    data_int3.z *= data.z;
                }else{
                    data_int3 = data;
                }
            } else if ( data_type_current == HELIOS_TYPE_INT4) {
                int4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if( i==0 ){
                    data_int4.x *= data.x;
                    data_int4.y *= data.y;
                    data_int4.z *= data.z;
                    data_int4.w *= data.w;
                }else{
                    data_int4 = data;
                }
            } else {
                helios_runtime_error("ERROR (Context::aggregatePrimitiveDataProduct): This operation is not supported for string primitive data types.");
            }
            i++;
        }

        if( !init_type ){
            primitive_data_not_exist++;
            continue;
        }else if ( data_type == HELIOS_TYPE_FLOAT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_float );
        } else if ( data_type == HELIOS_TYPE_DOUBLE) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_double );
        } else if ( data_type == HELIOS_TYPE_VEC2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec2 );
        } else if ( data_type == HELIOS_TYPE_VEC3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec3 );
        } else if ( data_type == HELIOS_TYPE_VEC4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_vec4 );
        } else if ( data_type == HELIOS_TYPE_INT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int );
        } else if ( data_type == HELIOS_TYPE_UINT) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_uint );
        } else if ( data_type == HELIOS_TYPE_INT2) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int2 );
        } else if ( data_type == HELIOS_TYPE_INT3) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int3 );
        } else if ( data_type == HELIOS_TYPE_INT4) {
            setPrimitiveData( UUID, result_primitive_data_label.c_str(), data_int4 );
        }

    }

    if( primitives_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataProduct): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if( primitive_data_not_exist>0 ){
        std::cout << "WARNING (Context::aggregatePrimitiveDataProduct): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no multiplication was performed and new primitive data was not created for this primitive." << std::endl;
    }

}


float Context::sumPrimitiveSurfaceArea( const std::vector<uint> &UUIDs ) const{

  bool primitive_warning = false;
  float area = 0;
  for( uint UUID : UUIDs ){

    if( doesPrimitiveExist(UUID) ){
      area += getPrimitiveArea(UUID);
    }else{
      primitive_warning = true;
    }

  }

  if( primitive_warning ){
    std::cout << "WARNING (Context::sumPrimitiveSurfaceArea): One or more primitives reference in the UUID vector did not exist.";
  }

  return area;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, float filter_value, const std::string &comparator ){

  if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
    helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
  }

  std::vector<uint> UUIDs_out = UUIDs;
  for( int p=UUIDs.size()-1; p>=0; p-- ){
    uint UUID = UUIDs_out.at(p);
    if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_FLOAT ){
      float data;
      getPrimitiveData(UUID,primitive_data_label.c_str(),data);
      if( comparator=="==" && data==filter_value ){
        continue;
      }else if ( comparator==">" && data>filter_value ) {
        continue;
      }else if ( comparator=="<" && data<filter_value ){
        continue;
      }else if ( comparator==">=" && data>=filter_value ){
        continue;
      }else if ( comparator=="<=" && data<=filter_value ){
        continue;
      }

      std::swap( UUIDs_out.at(p),UUIDs_out.back() );
      UUIDs_out.pop_back();
    }
  }

  return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, double filter_value, const std::string &comparator ){

  if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
    helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
  }

  std::vector<uint> UUIDs_out = UUIDs;
  for( int p=UUIDs.size()-1; p>=0; p-- ){
    uint UUID = UUIDs_out.at(p);
    if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_DOUBLE ){
      double data;
      getPrimitiveData(UUID,primitive_data_label.c_str(),data);
      if( comparator=="==" && data==filter_value ){
        continue;
      }else if ( comparator==">" && data>filter_value ) {
        continue;
      }else if ( comparator=="<" && data<filter_value ){
        continue;
      }else if ( comparator==">=" && data>=filter_value ){
        continue;
      }else if ( comparator=="<=" && data<=filter_value ){
        continue;
      }

      std::swap( UUIDs_out.at(p),UUIDs_out.back() );
      UUIDs_out.pop_back();
    }
  }

  return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, int filter_value, const std::string &comparator ){

  if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
    helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
  }

  std::vector<uint> UUIDs_out = UUIDs;
  for( int p=UUIDs.size()-1; p>=0; p-- ){
    uint UUID = UUIDs_out.at(p);
    if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_INT ){
      int data;
      getPrimitiveData(UUID,primitive_data_label.c_str(),data);
      if( comparator=="==" && data==filter_value ){
        continue;
      }else if ( comparator==">" && data>filter_value ) {
        continue;
      }else if ( comparator=="<" && data<filter_value ){
        continue;
      }else if ( comparator==">=" && data>=filter_value ){
        continue;
      }else if ( comparator=="<=" && data<=filter_value ){
        continue;
      }

      std::swap( UUIDs_out.at(p),UUIDs_out.back() );
      UUIDs_out.pop_back();
    }
  }

  return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, uint filter_value, const std::string &comparator ){

  if( comparator!="==" && comparator!=">" && comparator!="<" && comparator!=">=" && comparator!="<="  ){
    helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
  }

  std::vector<uint> UUIDs_out = UUIDs;
  for( int p=UUIDs.size()-1; p>=0; p-- ){
    uint UUID = UUIDs_out.at(p);
    if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_UINT ){
      uint data;
      getPrimitiveData(UUID,primitive_data_label.c_str(),data);
      if( comparator=="==" && data==filter_value ){
        continue;
      }else if ( comparator==">" && data>filter_value ) {
        continue;
      }else if ( comparator=="<" && data<filter_value ){
        continue;
      }else if ( comparator==">=" && data>=filter_value ){
        continue;
      }else if ( comparator=="<=" && data<=filter_value ){
        continue;
      }

      std::swap( UUIDs_out.at(p),UUIDs_out.back() );
      UUIDs_out.pop_back();
    }
  }

  return UUIDs_out;

}

std::vector<uint> Context::filterPrimitivesByData( const std::vector<uint> &UUIDs, const std::string &primitive_data_label, const std::string &filter_value ){

  std::vector<uint> UUIDs_out = UUIDs;
  for( int p=UUIDs.size()-1; p>=0; p-- ){
    uint UUID = UUIDs_out.at(p);
    if( doesPrimitiveDataExist(UUID,primitive_data_label.c_str()) && getPrimitiveDataType(UUID,primitive_data_label.c_str())==HELIOS_TYPE_STRING ){
      std::string data;
      getPrimitiveData(UUID,primitive_data_label.c_str(),data);
      if( data!=filter_value ) {
        std::swap(UUIDs_out.at(p), UUIDs_out.back());
        UUIDs_out.pop_back();
      }
    }
  }

  return UUIDs_out;

}