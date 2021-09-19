/** \file "global.cpp" global declarations. 
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

#include "global.h"

//PNG Libraries (reading and writing PNG images)
#include <unistd.h>
#define PNG_DEBUG 3
#define PNG_SKIP_SETJMP_CHECK 1
#include <png.h>

using namespace helios;

RGBcolor RGB::red = make_RGBcolor( 1.f, 0.f, 0.f );
RGBcolor RGB::blue = make_RGBcolor( 0.f, 0.f, 1.f );
RGBcolor RGB::green = make_RGBcolor( 0.f, 0.6f, 0.f );
RGBcolor RGB::cyan = make_RGBcolor( 0.f, 1.f, 1.f );
RGBcolor RGB::magenta = make_RGBcolor( 1.f, 0.f, 1.f );
RGBcolor RGB::yellow = make_RGBcolor( 1.f, 1.f, 0.f );
RGBcolor RGB::orange = make_RGBcolor( 1.f, 0.5f, 0.f );
RGBcolor RGB::violet = make_RGBcolor( 0.5f, 0.f, 0.5f );
RGBcolor RGB::black = make_RGBcolor( 0.f, 0.f, 0.f );
RGBcolor RGB::white = make_RGBcolor( 1.f, 1.f, 1.f );
RGBcolor RGB::lime = make_RGBcolor( 0.f, 1.f, 0.f );
RGBcolor RGB::silver = make_RGBcolor( 0.75f, 0.75f, 0.75f );
RGBcolor RGB::gray = make_RGBcolor( 0.5f, 0.5f, 0.5f );
RGBcolor RGB::navy = make_RGBcolor( 0.f, 0.f, 0.5f );
RGBcolor RGB::brown = make_RGBcolor( 0.55f, 0.27f, 0.075 );
RGBcolor RGB::khaki = make_RGBcolor( 0.94f, 0.92f, 0.55f );
RGBcolor RGB::greenyellow = make_RGBcolor( 0.678f, 1.f, 0.184f );
RGBcolor RGB::forestgreen = make_RGBcolor( 0.133f, 0.545f, 0.133f );
RGBcolor RGB::yellowgreen = make_RGBcolor( 0.6, 0.8, 0.2 );
RGBcolor RGB::goldenrod = make_RGBcolor( 0.855, 0.647, 0.126 );

RGBcolor helios::blend( RGBcolor color0, RGBcolor color1, float weight ){
  RGBcolor color;
  color.r = weight*color1.r+(1.f-weight)*color0.r;
  color.g = weight*color1.g+(1.f-weight)*color0.g;
  color.b = weight*color1.b+(1.f-weight)*color0.b;
  return color;
}

RGBAcolor helios::blend( RGBAcolor color0, RGBAcolor color1, float weight ){
  RGBAcolor color;
  color.r = weight*color1.r+(1.f-weight)*color0.r;
  color.g = weight*color1.g+(1.f-weight)*color0.g;
  color.b = weight*color1.b+(1.f-weight)*color0.b;
  color.a = weight*color1.a+(1.f-weight)*color0.a;
  return color;
}

vec3 helios::rotatePoint(const vec3 position, const SphericalCoord rotation ) {
  return rotatePoint( position, rotation.elevation, rotation.azimuth );
}

vec3 helios::rotatePoint(const vec3 position, const float theta, const float phi) {

  float Ry[3][3], Rz[3][3];

  float st = sin(theta);
  float ct = cos(theta);

  float sp = sin(phi);
  float cp = cos(phi);
	
  // Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
  Ry[0][0] = ct;
  Ry[0][1] = 0.f;
  Ry[0][2] = st;
  Ry[1][0] = 0.f;
  Ry[1][1] = 1.f;
  Ry[1][2] = 0.f;
  Ry[2][0] = -st;
  Ry[2][1] = 0.f;
  Ry[2][2] = ct;

  Rz[0][0] = cp;
  Rz[0][1] = -sp;
  Rz[0][2] = 0.f;
  Rz[1][0] = sp;
  Rz[1][1] = cp;
  Rz[1][2] = 0.f;
  Rz[2][0] = 0.f;
  Rz[2][1] = 0.f;
  Rz[2][2] = 1.f;

  // Multiply Ry*Rz

  float rotMat[3][3] = {0.f};

  for( int i=0;i<3;i++){
    for(int j=0;j<3;j++){
      for(int k=0;k<3;k++){
	rotMat[i][j]=rotMat[i][j]+Rz[i][k]*Ry[k][j];
      }
    }
  }
	
  // Multiply the rotation matrix with the position vector.
  vec3 tmp;
  tmp.x = rotMat[0][0] * position.x + rotMat[0][1] * position.y + rotMat[0][2] * position.z;
  tmp.y = rotMat[1][0] * position.x + rotMat[1][1] * position.y + rotMat[1][2] * position.z;
  tmp.z = rotMat[2][0] * position.x + rotMat[2][1] * position.y + rotMat[2][2] * position.z;
  
  return tmp;
	
}

vec3 helios::rotatePointAboutLine(const vec3 point, const vec3 line_base, const vec3 line_direction, const float theta) {

  //for reference this was taken from http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/

  vec3 position;

  vec3 tmp = line_direction;
  tmp.normalize();
  float u = tmp.x;
  float v = tmp.y;
  float w = tmp.z;

  float a = line_base.x;
  float b = line_base.y;
  float c = line_base.z;

  float x = point.x;
  float y = point.y;
  float z = point.z;

  float st = sin(theta);
  float ct = cos(theta);

  position.x = (a*(v*v+w*w)-u*(b*v+c*w-u*x-v*y-w*z))*(1-ct)+x*ct+(-c*v+b*w-w*y+v*z)*st;
  position.y = (b*(u*u+w*w)-v*(a*u+c*w-u*x-v*y-w*z))*(1-ct)+y*ct+(c*u-a*w+w*x-u*z)*st;
  position.z = (c*(u*u+v*v)-w*(a*u+b*v-u*x-v*y-w*z))*(1-ct)+z*ct+(-b*u+a*v-v*x+u*y)*st;
	
  return position;
	
}

float helios::calculateTriangleArea( const vec3 v0, const vec3 v1, const vec3 v2 ){
  vec3 A( v1-v0 );
  vec3 B( v2-v0 );
  vec3 C( v2-v1 );
  float a = A.magnitude();
  float b = B.magnitude();
  float c = C.magnitude();
  float s = 0.5f*( a+b+c );
  return sqrtf( s*(s-a)*(s-b)*(s-c) );
}

int helios::Date::JulianDay( void ) const{

  int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
  int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int* skips;
  
  if( (year-2000)%4 == 0 ){  //leap year
    skips=skips_leap;
  }else{                 //non-leap year
    skips=skips_nonleap;
  }
  
  return skips[month-1]+day;

}

float helios::randu( void ){

  return float(rand()) / float(RAND_MAX + 1.); 

}

int helios::randu( int imin, int imax  ){

  float ru = randu();

  if( imin==imax || imin>imax ){
    return imin;
  }else{
    return imin + round(float(imax-imin)*ru);
  }
    
}

float helios::acos_safe( float x ){
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return acosf(x) ;
}

float helios::asin_safe( float x ){
  if (x < -1.0) x = -1.0 ;
  else if (x > 1.0) x = 1.0 ;
  return asinf(x) ;
}

bool helios::lineIntersection( const helios::vec2 p1, const helios::vec2 q1, const helios::vec2 p2, const helios::vec2 q2)
{
 
    float ax = q1.x - p1.x;     // direction of line a
    float ay = q1.y - p1.y;     // ax and ay as above

    float bx = p2.x - q2.x;     // direction of line b, reversed
    float by = p2.y - q2.y;     // really -by and -by as above

    float dx = p2.x - p1.x;   // right-hand side
    float dy = p2.y - p1.y;

    float det = ax * by - ay * bx;

    if (det == 0) return false;

    float r = (dx * by - dy * bx) / det;
    float s = (ax * dy - ay * dx) / det;

    return !(r < 0 || r > 1 || s < 0 || s > 1);
    
}

bool helios::pointInPolygon( const helios::vec2 point, const std::vector<helios::vec2> polygon_verts ){

  std::vector<vec2> pverts = polygon_verts;
  pverts.push_back( polygon_verts.front() );

  vec2 p1 = point;
  vec2 q1 = make_vec2(100000,0); //todo: calculate a bounding box

  int Nintersect = 0;
  for( size_t i=0; i<pverts.size()-1; i++ ){

    vec2 p2 = pverts.at(i);
    vec2 q2 = pverts.at(i+1);

    bool isect = lineIntersection( p1, q1, p2, q2 );

    if( isect ){
      Nintersect ++;
    }
    
  }

  if( Nintersect!=0 && Nintersect%2==1 ){
    return true;
  }else{
    return false;
  }

}

void helios::wait( float seconds ){

  int msec = round( seconds*1000.f );
  std::this_thread::sleep_for(std::chrono::milliseconds(msec));

}

void helios::makeRotationMatrix( const float rotation, const char* axis, float (&T)[16] ){

  float sx = sin(rotation);
  float cx = cos(rotation);

  if( strcmp(axis,"x")==0 ){  
    T[0] = 1.f; //(0,0)
    T[1] = 0.f; //(0,1)
    T[2] = 0.f; //(0,2)
    T[3] = 0.f; //(0,3)
    T[4] = 0.f; //(1,0)
    T[5] = cx;  //(1,1)
    T[6] = -sx; //(1,2)
    T[7] = 0.f; //(1,3)
    T[8] = 0.f; //(2,0)
    T[9] = sx;  //(2,1)
    T[10] = cx; //(2,2)
    T[11] = 0.f;//(2,3)
  }else if( strcmp(axis,"y")==0 ){  
    T[0] = cx;  //(0,0)
    T[1] = 0.f; //(0,1)
    T[2] = sx;  //(0,2)
    T[3] = 0.f; //(0,3)
    T[4] = 0.f; //(1,0)
    T[5] = 1.f; //(1,1)
    T[6] = 0.f; //(1,2)
    T[7] = 0.f; //(1,3)
    T[8] = -sx; //(2,0)
    T[9] = 0.f; //(2,1)
    T[10] = cx; //(2,2)
    T[11] = 0.f;//(2,3)
 }else if( strcmp(axis,"z")==0 ){     
    T[0] = cx;  //(0,0)
    T[1] = -sx; //(0,1)
    T[2] = 0.f; //(0,2)
    T[3] = 0.f; //(0,3)
    T[4] = sx;  //(1,0)
    T[5] = cx;  //(1,1)
    T[6] = 0.f; //(1,2)
    T[7] = 0.f; //(1,3)
    T[8] = 0.f; //(2,0)
    T[9] = 0.f; //(2,1)
    T[10] = 1.f;//(2,2)
    T[11] = 0.f;//(2,3)
  }else{
    std::cerr << "ERROR (makeRotationMatrix): Rotation axis should be one of x, y, or z." << std::endl;
    exit(EXIT_FAILURE);
  }
  T[12]=T[13]=T[14]=0.f;
  T[15]=1.f;

}

void helios::makeRotationMatrix( const float rotation, const helios::vec3 axis, float (&T)[16] ){

  vec3 u = axis;
  u.normalize();

  float sx = sin(rotation);
  float cx = cos(rotation);

  T[0] = cx+u.x*u.x*(1.f-cx); //(0,0)
  T[1] = u.x*u.y*(1.f-cx)-u.z*sx; //(0,1)
  T[2] = u.x*u.z*(1.f-cx)+u.y*sx; //(0,2)
  T[3] = 0.f; //(0,3)
  T[4] = u.y*u.x*(1.f-cx)+u.z*sx; //(1,0)
  T[5] = cx+u.y*u.y*(1.f-cx);  //(1,1)
  T[6] = u.y*u.z*(1.f-cx)-u.x*sx; //(1,2)
  T[7] = 0.f; //(1,3)
  T[8] = u.z*u.x*(1.f-cx)-u.y*sx; //(2,0)
  T[9] = u.z*u.y*(1.f-cx)+u.x*sx;  //(2,1)
  T[10] = cx+u.z*u.z*(1.f-cx); //(2,2)
  T[11] = 0.f;//(2,3)

  T[12]=T[13]=T[14]=0.f;
  T[15]=1.f;

}

void helios::makeTranslationMatrix( const helios::vec3 translation, float (&T)[16] ){

  T[0] = 1.f; //(0,0)
  T[1] = 0.f; //(0,1)
  T[2] = 0.f; //(0,2)
  T[3] = translation.x; //(0,3)
  T[4] = 0.f; //(1,0)
  T[5] = 1.f;  //(1,1)
  T[6] = 0.f; //(1,2)
  T[7] = translation.y; //(1,3)
  T[8] = 0.f; //(2,0)
  T[9] = 0.f;  //(2,1)
  T[10] = 1.f; //(2,2)
  T[11] = translation.z;//(2,3)
  T[12] = 0.f;//(3,0)
  T[13] = 0.f;//(3,1)
  T[14] = 0.f;//(3,2)
  T[15] = 1.f;//(3,3)
  
}

void helios::makeScaleMatrix( const helios::vec3 scale, float (&T)[16] ){

  T[0] = scale.x; //(0,0)
  T[1] = 0.f; //(0,1)
  T[2] = 0.f; //(0,2)
  T[3] = 0.f; //(0,3)
  T[4] = 0.f; //(1,0)
  T[5] = scale.y;  //(1,1)
  T[6] = 0.f; //(1,2)
  T[7] = 0.f; //(1,3)
  T[8] = 0.f; //(2,0)
  T[9] = 0.f;  //(2,1)
  T[10] = scale.z; //(2,2)
  T[11] = 0.f;//(2,3)
  T[12] = 0.f;//(3,0)
  T[13] = 0.f;//(3,1)
  T[14] = 0.f;//(3,2)
  T[15] = 1.f;//(3,3)
  
}

void helios::matmult( const float ML[16], const float MR[16], float (&T)[16] ){

  float M[16]={0.f};

  for( int i=0;i<4;i++){
    for(int j=0;j<4;j++){
      for(int k=0;k<4;k++){
	M[4*i+j]=M[4*i+j]+ML[4*i+k]*MR[4*k+j];
      }
    }
  }

  for( int i=0;i<16;i++){
    T[i]=M[i];
  }

}

void helios::vecmult( const float M[16], const helios::vec3 v3, helios::vec3& result ){

  float v[4];
  v[0] = v3.x;
  v[1] = v3.y;
  v[2] = v3.z;
  v[3] = 1.f;

  float V[4]={0.f};

  for(int j=0;j<4;j++){
    for(int k=0;k<4;k++){
      V[j]=V[j]+v[k]*M[k+4*j];
    }
  }

  result.x = V[0];
  result.y = V[1];
  result.z = V[2];
  
}

void helios::vecmult( const float M[16], const float v[3], float (&result)[3] ){

  float V[4]={0.f};

  for(int j=0;j<4;j++){
    for(int k=0;k<4;k++){
      V[j]=V[j]+v[k]*M[k+4*j];
    }
  }

  for( int i=0;i<3;i++){
    result[i]=V[i];
  }

}

void helios::makeIdentityMatrix( float (&T)[16] ){

  /* [0,0] */ T[0] = 1.f; 
  /* [0,1] */ T[1] = 0.f;
  /* [0,2] */ T[2] = 0.f;
  /* [0,3] */ T[3] = 0.f;
  /* [1,0] */ T[4] = 0.f;
  /* [1,1] */ T[5] = 1.f;
  /* [1,2] */ T[6] = 0.f;
  /* [1,3] */ T[7] = 0.f;
  /* [2,0] */ T[8] = 0.f;
  /* [2,1] */ T[9] = 0.f;
  /* [2,2] */ T[10] = 1.f;
  /* [2,3] */ T[11] = 0.f;
  /* [3,0] */ T[12] = 0.f;
  /* [3,1] */ T[13] = 0.f;
  /* [3,2] */ T[14] = 0.f;
  /* [3,3] */ T[15] = 1.f;

}

// void helios::makeTransformationMatrix( const vec3 anchor, const vec3 size, float (&transform)[16] ){

//   float Rx[3][3], Ry[3][3], Rz[3][3];

//   //It causes problems if size=0
//   vec3 sz = size;
//   if( size.x == 0 ){
//     sz.x = 1.f;
//   }else if( size.y == 0 ){
//     sz.y = 1.f;
//   }else if( size.z == 0 ){
//     sz.z = 1.f;
//   }

//   // float sx = sin(rotation.x);
//   // float cx = cos(rotation.x);

//   // float sy = sin(rotation.y);
//   // float cy = cos(rotation.y);

//   // float sz = sin(rotation.z);
//   // float cz = cos(rotation.z);
	
//   // // Setup the rotation matrix, this matrix is based off of the rotation matrix used in glRotatef.
//   // Rx[0][0] = 1.f*size.x;
//   // Rx[0][1] = 0.f;
//   // Rx[0][2] = 0.f;
//   // Rx[1][0] = 0.f;
//   // Rx[1][1] = cx*size.y;
//   // Rx[1][2] = -sx*size.z;
//   // Rx[2][0] = 0.f;
//   // Rx[2][1] = sx*size.y;
//   // Rx[2][2] = cx*size.z;

//   // Ry[0][0] = cy;
//   // Ry[0][1] = 0.f;
//   // Ry[0][2] = sy;
//   // Ry[1][0] = 0.f;
//   // Ry[1][1] = 1.f;
//   // Ry[1][2] = 0.f;
//   // Ry[2][0] = -sy;
//   // Ry[2][1] = 0.f;
//   // Ry[2][2] = cy;

//   // Rz[0][0] = cz;
//   // Rz[0][1] = -sz;
//   // Rz[0][2] = 0.f;
//   // Rz[1][0] = sz;
//   // Rz[1][1] = cz;
//   // Rz[1][2] = 0.f;
//   // Rz[2][0] = 0.f;
//   // Rz[2][1] = 0.f;
//   // Rz[2][2] = 1.f;

//   // float R[4][4]={0.f};
//   // float Rtemp[4][4]={0.f};

//   // // Multiply Rx*Ry
//   // for( int i=0;i<3;i++){
//   //   for(int j=0;j<3;j++){
//   //     for(int k=0;k<3;k++){
//   // 	Rtemp[i][j]=Rtemp[i][j]+Ry[i][k]*Rx[k][j];
//   //     }
//   //   }
//   // }

//   // // Multiply Rtemp*Rz
//   // for( int i=0;i<3;i++){
//   //   for(int j=0;j<3;j++){
//   //     for(int k=0;k<3;k++){
//   // 	R[i][j]=R[i][j]+Rz[i][k]*Rtemp[k][j];
//   //     }
//   //   }
//   // }

//   // R[3][3] = 1.f;

//   // // Translation Matrix

//   // float T[4][4]={0.f};
  
//   // T[0][0] = T[1][1] = T[2][2] = T[3][3] = 1.f;

//   // T[0][3] = anchor.x;
//   // T[1][3] = anchor.y;
//   // T[2][3] = anchor.z;

//   // // Multiply T*R (should rotate about origin first, then translate)

//   // float M[4][4]={0.f};

//   // for( int i=0;i<4;i++){
//   //   for(int j=0;j<4;j++){
//   //     for(int k=0;k<4;k++){
//   // 	M[i][j]=M[i][j]+T[i][k]*R[k][j];
//   //     }
//   //   }
//   // }


//   // // Rearrange 4x4 Matrix into 16x1 Array
//   // /* [0,0] */ transform[0] =  M[0][0];
//   // /* [0,1] */ transform[1] =  M[0][1];
//   // /* [0,2] */ transform[2] =  M[0][2];
//   // /* [0,3] */ transform[3] =  M[0][3];
//   // /* [1,0] */ transform[4] =  M[1][0];
//   // /* [1,1] */ transform[5] =  M[1][1];
//   // /* [1,2] */ transform[6] =  M[1][2];
//   // /* [1,3] */ transform[7] =  M[1][3];
//   // /* [2,0] */ transform[8] =  M[2][0];
//   // /* [2,1] */ transform[9] =  M[2][1];
//   // /* [2,2] */ transform[10] = M[2][2];
//   // /* [2,3] */ transform[11] = M[2][3];
//   // /* [3,0] */ transform[12] = M[3][0];
//   // /* [3,1] */ transform[13] = M[3][1];
//   // /* [3,2] */ transform[14] = M[3][2];
//   // /* [3,3] */ transform[15] = M[3][3];

//   /* [0,0] */ transform[0] = sz.x; 
//   /* [0,1] */ transform[1] = 0.f;
//   /* [0,2] */ transform[2] = 0.f;
//   /* [0,3] */ transform[3] = anchor.x;
//   /* [1,0] */ transform[4] = 0.f;
//   /* [1,1] */ transform[5] = sz.y;
//   /* [1,2] */ transform[6] =  0.f;
//   /* [1,3] */ transform[7] = anchor.y;
//   /* [2,0] */ transform[8] = 0.f;
//   /* [2,1] */ transform[9] = 0.f;
//   /* [2,2] */ transform[10] = sz.z;
//   /* [2,3] */ transform[11] = anchor.z;
//   /* [3,0] */ transform[12] = 0.f;
//   /* [3,1] */ transform[13] = 0.f;
//   /* [3,2] */ transform[14] = 0.f;
//   /* [3,3] */ transform[15] = 1.f;

// }

float helios::deg2rad( const float& deg ){
  return deg*M_PI/180.f;
}

float helios::rad2deg( const float& rad ){
  return rad*180.f/M_PI;
}

float helios::atan2_2pi( const float& y, const float& x){

  float v=0;

  if(x>0.f){
    v=atan(y/x);
  }
  if(y>=0.f && x<0.f){
    v=M_PI+atan(y/x);
  }
  if(y<0.f && x<0.f){
    v=-M_PI+atan(y/x);
  }
  if(y>0.f && x==0.f){
    v=0.5f*M_PI;
  }
  if(y<0.f && x==0.f){
    v=-0.5f*M_PI;
  }
  if(v<0.f){
    v=v+2.f*M_PI;
  }
  return v;

}

SphericalCoord helios::cart2sphere( const vec3& Cartesian ){

  SphericalCoord Spherical;

  Spherical.radius = sqrt( Cartesian.x*Cartesian.x + Cartesian.y*Cartesian.y + Cartesian.z*Cartesian.z );

  Spherical.elevation = asin( Cartesian.z/Spherical.radius );

  Spherical.zenith = 0.5f*M_PI - Spherical.elevation;

  Spherical.azimuth = atan2_2pi( Cartesian.x, Cartesian.y );

  return Spherical;
  
}

vec3 helios::sphere2cart( const SphericalCoord& Spherical ){

  vec3 Cartesian;

  Cartesian.x = Spherical.radius*cos(Spherical.elevation)*sin(Spherical.azimuth);
  Cartesian.y = Spherical.radius*cos(Spherical.elevation)*cos(Spherical.azimuth);
  Cartesian.z = Spherical.radius*sin(Spherical.elevation);

  return Cartesian;

}

vec2 helios::string2vec2( const char* str ){

  float o[2] = {99999,99999};
  std::string tmp;
  
  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==2 ){
      break;
    }
    o[c] = stof(tmp);
    c++;
  }
  return make_vec2( o[0], o[1] );

}

vec3 helios::string2vec3( const char* str ){

  float o[3]  = {99999,99999,99999};
  std::string tmp;
  
  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==3 ){
      break;
    }
    o[c] = stof(tmp);
    c++;
  }
  return make_vec3( o[0], o[1], o[2] );

}

vec4 helios::string2vec4( const char* str ){

  float o[4]  = {99999,99999,99999,99999};
  std::string tmp;
  
  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==4 ){
      break;
    }
    o[c] = stof(tmp);
    c++;
  }
  return make_vec4( o[0], o[1], o[2], o[3] );

}

int2 helios::string2int2( const char* str ){

  int o[2] = {99999,99999};
  std::string tmp;

  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==2 ){
      break;
    }
    o[c] = stoi(tmp);
    c++;
  }
  return make_int2( o[0], o[1] );

}

int3 helios::string2int3( const char* str ){

  int o[3] = {99999,99999,99999};
  std::string tmp;
  
  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==3 ){
      break;
    }
    o[c] = stoi(tmp);
    c++;
  }
  return make_int3( o[0], o[1], o[2] );

}

int4 helios::string2int4( const char* str ){

  int o[4] = {99999,99999,99999,99999};
  std::string tmp;
  
  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==4 ){
      break;
    }
    o[c] = stoi(tmp);
    c++;
  }
  return make_int4( o[0], o[1], o[2], o[3] );

}

RGBAcolor helios::string2RGBcolor( const char* str ){

  float o[4] = {0,0,0,1};
  std::string tmp;
  
  std::istringstream stream(str);
  int c=0;
  while( stream >> tmp ){
    if( c==4 ){
      break;
    }
    o[c] = stof(tmp);
    c++;
  }
  return make_RGBAcolor( o[0], o[1], o[2], o[3] );

}


const char* helios::deblank(const char* input)                                         
{
    int i,j;
    char* output = strdup(input);
    for (i = 0, j = 0; i<strlen(input); i++,j++)          
    {
        if (input[i]!=' ')                           
            output[j]=input[i];                     
        else
            j--;                                     
    }
    output[j]=0;
    const char* output_c = output;
    return output_c;
}

template <typename anytype>
anytype helios::clamp( anytype value, anytype min, anytype max ){
  if( value<min ){
    value = min;
  }else if( value>max ){
    value = max;
  }
  return value;
}

float helios::mean( std::vector<float> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (mean): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  float m = 0;
  for( uint i=0; i<vect.size(); i++ ){
    m += vect.at(i);
  }
  m /= float(vect.size());

  return m;

}

float helios::min( std::vector<float> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (min): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  return *std::min_element(vect.begin(),vect.end());

}

int helios::min( std::vector<int> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (min): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  return *std::min_element(vect.begin(),vect.end());

}

vec3 helios::min( std::vector<vec3> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (min): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  vec3 vmin = vect.at(0);

  for( int i=1; i<vect.size(); i++ ){

    if( vect.at(i).x < vmin.x ){
      vmin.x = vect.at(i).x;
    }
    if( vect.at(i).y < vmin.y ){
      vmin.y = vect.at(i).y;
    }
    if( vect.at(i).z < vmin.z ){
      vmin.z = vect.at(i).z;
    }
    
  }

  return vmin;

}

float helios::max( std::vector<float> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (max): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  return *std::max_element(vect.begin(),vect.end());

}

int helios::max( std::vector<int> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (max): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  return *std::max_element(vect.begin(),vect.end());

}

vec3 helios::max( std::vector<vec3> vect ){

  if( vect.size()==0 ){
    std::cerr << "ERROR (max): Vector is empty." << std::endl;
    exit(EXIT_FAILURE);
  }

  vec3 vmax = vect.at(0);

  for( int i=1; i<vect.size(); i++ ){

    if( vect.at(i).x > vmax.x ){
      vmax.x = vect.at(i).x;
    }
    if( vect.at(i).y > vmax.y ){
      vmax.y = vect.at(i).y;
    }
    if( vect.at(i).z > vmax.z ){
      vmax.z = vect.at(i).z;
    }
    
  }

  return vmax;

}

template <typename anytype>
void resize_vector( std::vector<std::vector<anytype> > &vec, uint Nx, uint Ny ){
  vec.resize(Ny);
  for( int j=0; j<Ny; j++ ){
    vec.at(j).resize(Nx);
  }
}

template <typename anytype>
void resize_vector( std::vector<std::vector<std::vector<anytype> > > &vec, int Nx, int Ny, int Nz ){
  vec.resize(Nz);
  for( int k=0; k<Nz; k++ ){
    vec.at(k).resize(Ny);
    for( int j=0; j<Ny; j++ ){
      vec.at(k).at(j).resize(Nx);
    }
  }
}

template <typename anytype>
void resize_vector( std::vector<std::vector<std::vector<std::vector<anytype> > > > &vec, int Nx, int Ny, int Nz, int Nw ){
  vec.resize(Nw);
  for( int w=0; w<Nw; w++ ){
    vec.at(w).resize(Nz);
    for( int k=0; k<Nz; k++ ){
      vec.at(w).at(k).resize(Ny);
      for( int j=0; j<Ny; j++ ){
	vec.at(w).at(k).at(j).resize(Nx);
      }
    }
  }
}

Date helios::CalendarDay( int Julian_day, int year ){

  if( Julian_day<1 || Julian_day>366 ){
    std::cerr << "ERROR (CalendarDay): Julian day out of range." << std::endl;
    exit(EXIT_FAILURE);
  }else if( year<1000 ){
    std::cerr << "ERROR (CalendarDay): Year should be specified in YYYY format." << std::endl;
    exit(EXIT_FAILURE);
  }

  int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 214, 244, 274, 305, 335};
  int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int* skips;
  int day, month;
    
  if( (year-2000)%4 == 0 ){  //leap year
    skips=skips_leap;
  }else{                 //non-leap year
    skips=skips_nonleap;
  }
  
  //set month
  int i;
  for( i=0; i<12; i++ ){
    if(i==11){
      month=12;
      break;
    }else if(Julian_day>skips[i] && Julian_day<=skips[i+1]){
      month=i+1;
      break;
    }
  }
  
  //set day
  if(skips[i]==0){
    day=Julian_day;
  }else{
    day=Julian_day%skips[i];
  }

  return make_Date(day,month,year);
  
}

int helios::JulianDay( int day, int month, int year ){

  return JulianDay( make_Date(day,month,year) );
    
}

int helios::JulianDay( Date date ){

  int day = date.day;
  int month = date.month;
  int year = date.year;

  if( day<1 || day>31 ){
    std::cerr << "ERROR (JulianDay): Day of month is out of range (day of " << day << " was given)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( month<1 || month>12){
    std::cerr << "ERROR (JulianDay): Month of year is out of range (month of " << month << " was given)." << std::endl;
    exit(EXIT_FAILURE);
  }else if( year<1000 ){
    std::cerr << "ERROR (JulianDay): Year should be specified in YYYY format." << std::endl;
    exit(EXIT_FAILURE);
  }

  int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 214, 244, 274, 305, 335};
  int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int* skips;
  
  if( (year-2000)%4 == 0 ){  //leap year
    skips=skips_leap;
  }else{                 //non-leap year
    skips=skips_nonleap;
  }
  
  return skips[month]+day;
    
}

// void Glyph::readFile( const char* __filename ){

//   filename = (char*)__filename;

//   //read the mask file
//   std::cout << "Reading mask file: " << filename << "..." << std::flush;
//   std::ifstream file (filename); //open the file
//   if(!file.is_open()){ //check that file exists
//     std::cout << "failed." << std::endl;
//     std::cerr << "ERROR: mask file does not exist." << std::endl;
//     exit(EXIT_FAILURE);
//   }
  
//   file >> size.x; //read the width of mask from the header
//   file >> size.y; //read the height of mask from the header
  
//   data.resize(size.y);

//   float temp;
  
//   int solid_count = -1;
//   for( int j=0; j<size.y; j++ ){
//     data.at(j).resize(size.x);
//     for( int i=0; i<size.x; i++ ){
//       file >> temp;
//       data.at(j).at(i) = bool(temp);
//       if( data.at(j).at(i) == 1 ){
// 	solid_count++;
// 	// }else if( data.at(j).at(i) !=0 ){
// 	// 	std::cout << "WARNING (polyMask): Mask file should only contain binary 0 or 1 values. Found value of " << data.at(j).at(i) << " which is automatically being set to 1." << std::endl;
// 	// 	data.at(j).at(i) = 1;
//       }
//     }
//   }
  
//   file.close();

//   if( solid_count == -1 ){
//     std::cout << "ERROR (Glyph::readFile): Mask is empty (all values in the mask were 0)." << std::endl;
//     exit(EXIT_FAILURE);
//   }else{
//     std::cout << "done." << std::endl;
//   }

//   solidFraction = float(solid_count)/float(size.x*size.y);

//   return;

// }

bool helios::PNGHasAlpha( const char* filename ){

  uint Nchannels;
  
  png_structp png_ptr;
  png_infop info_ptr;

  char header[8]; 

  /* open file and test for it being a png */
  FILE *fp = fopen(filename, "rb");
  if (!fp){
    std::cerr << "ERROR (readPNGAlpha): File " << filename << " could not be opened for reading. The file either does not exist or you do not have permission to read it." << std::endl;
    exit(EXIT_FAILURE);
  }
  fread(header, 1, 8, fp);
  
  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr){
    std::cerr << "ERROR (read_png_alpha): png_create_read_struct failed." << std::endl;
    exit(EXIT_FAILURE);
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr){
    std::cerr << "ERROR (read_png_alpha): png_create_info_struct failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  
  if (setjmp(png_jmpbuf(png_ptr))){
    std::cerr << "ERROR (read_png_alpha): init_io failed." << std::endl;
    exit(EXIT_FAILURE);
  }  

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  
  png_read_info(png_ptr, info_ptr);

  Nchannels = png_get_channels(png_ptr, info_ptr);

  if( Nchannels==4 ){
    return true;
  }else{
    return false;
  }

}

std::vector<std::vector<bool> > helios::readPNGAlpha( const char* filename ){

  int x, y;
  uint height, width;

  std::vector<std::vector<bool> > mask;

  png_byte color_type;
  png_byte bit_depth;
  
  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep * row_pointers;

  char header[8];    // 8 is the maximum size that can be checked

  /* open file and test for it being a png */
  FILE *fp = fopen(filename, "rb");
  if (!fp){
    std::cerr << "ERROR (readPNGAlpha): File " << filename << " could not be opened for reading." << std::endl;
    exit(EXIT_FAILURE);
  }
  fread(header, 1, 8, fp);
  // if (png_sig_cmp(header, 0, 8)){
  //   std::cerr << "ERROR (read_png_alpha): File " << filename << " is not recognized as a PNG file." << std::endl;
  //   exit(EXIT_FAILURE);
  // }

  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);

  if (!png_ptr){
    std::cerr << "ERROR (read_png_alpha): png_create_read_struct failed." << std::endl;
    exit(EXIT_FAILURE);
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr){
    std::cerr << "ERROR (read_png_alpha): png_create_info_struct failed." << std::endl;
    exit(EXIT_FAILURE);
  }
  
  if (setjmp(png_jmpbuf(png_ptr))){
    std::cerr << "ERROR (read_png_alpha): init_io failed." << std::endl;
    exit(EXIT_FAILURE);
  }  

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  
  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  mask.resize( height );
  for( uint i=0; i<height; i++ ){
    mask.at(i).resize(width);
  }

  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  if (setjmp(png_jmpbuf(png_ptr))){
    std::cerr << "ERROR (read_png_alpha): read_image failed." << std::endl;
    exit(EXIT_FAILURE);
  }

  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  for (y=0; y<height; y++)
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

  png_read_image(png_ptr, row_pointers);

  fclose(fp);

  for (uint j=0; j<height; j++){
    png_byte* row=row_pointers[j];
    for (int i=0; i < width; i++ ){
      png_byte* ba=&(row[i*4]);
      float alpha = ba[3];
      if( alpha<250 ){
	mask.at(j).at(i) = false;
      }else{
	mask.at(j).at(i) = true;
      } 
    }
  }

  free(row_pointers);

  return mask;
  
}

std::vector<int> helios::flatten( const std::vector<std::vector<int> > vec ){

  std::vector<int> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<uint> helios::flatten( const std::vector<std::vector<uint> > vec ){

  std::vector<uint> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<float> helios::flatten( const std::vector<std::vector<float> > vec ){

  std::vector<float> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<double> helios::flatten( const std::vector<std::vector<double> > vec ){

  std::vector<double> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<helios::vec2> helios::flatten( const std::vector<std::vector<helios::vec2> > vec ){

  std::vector<helios::vec2> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<helios::vec3> helios::flatten( const std::vector<std::vector<helios::vec3> > vec ){

  std::vector<helios::vec3> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<helios::vec4> helios::flatten( const std::vector<std::vector<helios::vec4> > vec ){

  std::vector<helios::vec4> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<helios::int2> helios::flatten( const std::vector<std::vector<helios::int2> > vec ){

  std::vector<helios::int2> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<helios::int3> helios::flatten( const std::vector<std::vector<helios::int3> > vec ){

  std::vector<helios::int3> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<helios::int4> helios::flatten( const std::vector<std::vector<helios::int4> > vec ){

  std::vector<helios::int4> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<std::string> helios::flatten( const std::vector<std::vector<std::string> > vec ){

  std::vector<std::string> flat;
  for( size_t i=0; i<vec.size(); i++ ){
    flat.insert( flat.end(), vec.at(i).begin(), vec.at(i).end() );
  }
  return flat;
  
}

std::vector<int> helios::flatten( const std::vector<std::vector<std::vector<int> > > vec ){

  std::vector<int> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<uint> helios::flatten( const std::vector<std::vector<std::vector<uint> > > vec ){

  std::vector<uint> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
 
}

std::vector<float> helios::flatten( const std::vector<std::vector<std::vector<float> > > vec ){

  std::vector<float> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<double> helios::flatten( const std::vector<std::vector<std::vector<double> > > vec ){

  std::vector<double> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<helios::vec2> helios::flatten( const std::vector<std::vector<std::vector<helios::vec2> > > vec ){

  std::vector<helios::vec2> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<helios::vec3> helios::flatten( const std::vector<std::vector<std::vector<helios::vec3> > > vec ){

  std::vector<helios::vec3> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<helios::vec4> helios::flatten( const std::vector<std::vector<std::vector<helios::vec4> > > vec ){

  std::vector<helios::vec4> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<helios::int2> helios::flatten( const std::vector<std::vector<std::vector<helios::int2> > > vec ){

  std::vector<helios::int2> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<helios::int3> helios::flatten( const std::vector<std::vector<std::vector<helios::int3> > > vec ){

  std::vector<helios::int3> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<helios::int4> helios::flatten( const std::vector<std::vector<std::vector<helios::int4> > > vec ){

  std::vector<helios::int4> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

std::vector<std::string> helios::flatten( const std::vector<std::vector<std::vector<std::string> > > vec ){

  std::vector<std::string> flat;
  for( size_t j=0; j<vec.size(); j++ ){
    for( size_t i=0; i<vec.at(j).size(); i++ ){
      flat.insert( flat.end(), vec.at(j).at(i).begin(), vec.at(j).at(i).end() );
    }
  }
  return flat;
  
}

helios::vec3 helios::spline_interp3( const float _u, const helios::vec3 x_start, const helios::vec3 tan_start, const helios::vec3 x_end, const helios::vec3 tan_end ){

  //Perform interpolation between two 3D points using Cubic Hermite Spline

  float u = _u;
  if( u<0 || u>1.f ){
    std::cout << "WARNING (spline_interp3): Clamping query point 'u' to the interval (0,1)" << std::endl;
    u = clamp( u, 0.f, 1.f );
  }

  //Basis matrix
  float B[16] = {2.f, -2.f, 1.f, 1.f, -3.f, 3.f, -2.f, -1.f, 0, 0, 1.f, 0, 1.f, 0, 0, 0};

  //Control matrix
  float C[12] = {x_start.x, x_start.y, x_start.z, x_end.x, x_end.y, x_end.z, tan_start.x, tan_start.y, tan_start.z, tan_end.x, tan_end.y, tan_end.z};

  //Parameter vector
  float P[4] = {u*u*u, u*u, u, 1.f};

  float R[12]={0.f};

  for( int i=0;i<4;i++){
    for(int j=0;j<3;j++){
      for(int k=0;k<4;k++){
	R[3*i+j]=R[3*i+j]+B[4*i+k]*C[3*k+j];
      }
    }
  }

  float xq[3]={0.f};

  for(int j=0;j<3;j++){
    for(int k=0;k<4;k++){
      xq[j]=xq[j]+P[k]*R[3*k+j];
    }
  }

  return make_vec3(xq[0],xq[1],xq[2]);

}

float helios::XMLloadfloat( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  float value;
  if( strlen(field_str)==0 ){
    value = 99999;
  }else{
    value = atof( field_str ); //note: pugi loads xml data as a character.  need to separate it into float
  }
  
  return value;
  
}

int helios::XMLloadint( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  int value;
  if( strlen(field_str)==0 ){
    value = 99999;
  }else{
    value = atoi( field_str ); //note: pugi loads xml data as a character.  need to separate it into int
  }
  
  return value;
  
}

std::string helios::XMLloadstring( pugi::xml_node node, const char* field ){

  const char* field_str = deblank(node.child_value(field));
    
  std::string value;
  if( strlen(field_str)==0 ){
    value = "99999";
  }else{
    value = field_str; //note: pugi loads xml data as a character.  need to separate it into int
  }
  
  return value;
  
}

helios::vec2 helios::XMLloadvec2( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::vec2 value;
  if( strlen(field_str)==0 ){
    value = make_vec2(99999,99999);
  }else{
    value = string2vec2( field_str ); //note: pugi loads xml data as a character.  need to separate it into 2 floats
  }
  
  return value;
  
}

helios::vec3 helios::XMLloadvec3( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::vec3 value;
  if( strlen(field_str)==0 ){
    value = make_vec3(99999,99999,99999);
  }else{
    value = string2vec3( field_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
  }
  
  return value;
  
}

helios::vec4 helios::XMLloadvec4( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::vec4 value;
  if( strlen(field_str)==0 ){
    value = make_vec4(99999,99999,99999,99999);
  }else{
    value = string2vec4( field_str ); //note: pugi loads xml data as a character.  need to separate it into 4 floats
  }
  
  return value;
  
}

helios::int2 helios::XMLloadint2( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::int2 value;
  if( strlen(field_str)==0 ){
    value = make_int2(99999,99999);
  }else{
    value = string2int2( field_str ); //note: pugi loads xml data as a character.  need to separate it into 2 ints
  }
  
  return value;
  
}

helios::int3 helios::XMLloadint3( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::int3 value;
  if( strlen(field_str)==0 ){
    value = make_int3(99999,99999,99999);
  }else{
    value = string2int3( field_str ); //note: pugi loads xml data as a character.  need to separate it into 3 ints
  }
  
  return value;
  
}

helios::int4 helios::XMLloadint4( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::int4 value;
  if( strlen(field_str)==0 ){
    value = make_int4(99999,99999,99999,99999);
  }else{
    value = string2int4( field_str ); //note: pugi loads xml data as a character.  need to separate it into 4 ints
  }
  
  return value;
  
}

helios::RGBcolor helios::XMLloadrgb( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::RGBAcolor value;
  if( strlen(field_str)==0 ){
    value = make_RGBAcolor(99999,99999,99999,99999);
  }else{
    value = string2RGBcolor( field_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
  }
  
  return make_RGBcolor(value.r,value.g,value.b);
  
}

helios::RGBAcolor helios::XMLloadrgba( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::RGBAcolor value;
  if( strlen(field_str)==0 ){
    value = make_RGBAcolor(99999,99999,99999,99999);
  }else{
    value = string2RGBcolor( field_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
  }
  
  return value;
  
}
