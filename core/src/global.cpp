/** \file "global.cpp" global declarations.

    Copyright (C) 2016-2024 Brian Bailey

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
#define PNG_DEBUG 3
#define PNG_SKIP_SETJMP_CHECK 1
#include <png.h>

//JPEG Libraries (reading and writing JPEG images)
#include <cstdio> //<-- note libjpeg requires this header be included before its headers.
#include <jpeglib.h>

using namespace helios;

void helios::helios_runtime_error( const std::string &error_message ){
#ifdef HELIOS_DEBUG
    std::cerr << error_message << std::endl;
#endif
    throw(std::runtime_error(error_message));
}

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

SphericalCoord helios::nullrotation = make_SphericalCoord(0,0);
vec3 helios::nullorigin = make_vec3(0,0,0);

RGBcolor helios::blend( RGBcolor color0, RGBcolor color1, float weight ){
  RGBcolor color;
  weight = clamp(weight,0.f,1.f);
  color.r = weight*color1.r+(1.f-weight)*color0.r;
  color.g = weight*color1.g+(1.f-weight)*color0.g;
  color.b = weight*color1.b+(1.f-weight)*color0.b;
  return color;
}

RGBAcolor helios::blend(const RGBAcolor &color0, const RGBAcolor &color1, float weight ){
  RGBAcolor color;
  weight = clamp(weight,0.f,1.f);
  color.r = weight*color1.r+(1.f-weight)*color0.r;
  color.g = weight*color1.g+(1.f-weight)*color0.g;
  color.b = weight*color1.b+(1.f-weight)*color0.b;
  color.a = weight*color1.a+(1.f-weight)*color0.a;
  return color;
}

vec3 helios::rotatePoint(const vec3& position, const SphericalCoord& rotation ) {
  return rotatePoint( position, rotation.elevation, rotation.azimuth );
}

vec3 helios::rotatePoint(const vec3& position, float theta, float phi) {

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

vec3 helios::rotatePointAboutLine( const vec3& point, const vec3& line_base, const vec3& line_direction, float theta) {

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

float helios::calculateTriangleArea( const vec3& v0, const vec3& v1, const vec3& v2 ){
  vec3 A( v1-v0 );
  vec3 B( v2-v0 );
  vec3 C( v2-v1 );
  float a = A.magnitude();
  float b = B.magnitude();
  float c = C.magnitude();
  float s = 0.5f*( a+b+c );
  return sqrtf( s*(s-a)*(s-b)*(s-c) );
}

int helios::Date::JulianDay() const{

  int skips_leap[] = {0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335};
  int skips_nonleap[] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334};
  int* skips;

  if( isLeapYear() ){  //leap year
    skips=skips_leap;
  }else{                 //non-leap year
    skips=skips_nonleap;
  }
  
  return skips[month-1]+day;

}

void helios::Date::incrementDay() {
    int JD = Calendar2Julian( *this );
    if( JD<365 || ( JD==356 && isLeapYear() ) ) {
        Date cal = Julian2Calendar(JD + 1, year);
        day = cal.day;
        month = cal.month;
    }else{ //it is last day of the year
        day = 1;
        month = 1;
        year ++;
    }
}

bool helios::Date::isLeapYear() const {
    if( (year-2000)%4 == 0 ){  //leap year
        return true;
    }else{                 //non-leap year
        return false;
    }
}

float helios::randu(){

  return float(rand()) / float(RAND_MAX + 1.); 

}

int helios::randu( int imin, int imax  ){

  float ru = randu();

  if( imin==imax || imin>imax ){
    return imin;
  }else{
    return imin + (int)lround(float(imax-imin)*ru);
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

bool helios::lineIntersection(const vec2 &p1, const vec2 &q1, const vec2 &p2, const vec2 &q2)
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

bool helios::pointInPolygon(const vec2 &point, const std::vector<vec2> &polygon_verts ){

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

  int msec = (int)lround( seconds*1000.f );
  std::this_thread::sleep_for(std::chrono::milliseconds(msec));

}

void helios::makeRotationMatrix( float rotation, const char* axis, float (&T)[16] ){

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
    helios_runtime_error("ERROR (makeRotationMatrix): Rotation axis should be one of x, y, or z.");
  }
  T[12]=T[13]=T[14]=0.f;
  T[15]=1.f;

}

void helios::makeRotationMatrix( float rotation, const helios::vec3& axis, float (&T)[16] ){

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

void helios::makeRotationMatrix( float rotation, const helios::vec3& origin, const helios::vec3& axis, float (&T)[16] ){

    //Construct inverse translation matrix to translate back to the origin
    float Ttrans[16];
    makeIdentityMatrix(Ttrans);

    Ttrans[3] = -origin.x; //(0,3)
    Ttrans[7] = -origin.y; //(1,3)
    Ttrans[11] = -origin.z;//(2,3)

    //Construct rotation matrix
    vec3 u = axis;
    u.normalize();

    float sx = sin(rotation);
    float cx = cos(rotation);

    float Trot[16];
    makeIdentityMatrix(Trot);

    Trot[0] = cx+u.x*u.x*(1.f-cx); //(0,0)
    Trot[1] = u.x*u.y*(1.f-cx)-u.z*sx; //(0,1)
    Trot[2] = u.x*u.z*(1.f-cx)+u.y*sx; //(0,2)
    Trot[3] = 0.f; //(0,3)
    Trot[4] = u.y*u.x*(1.f-cx)+u.z*sx; //(1,0)
    Trot[5] = cx+u.y*u.y*(1.f-cx);  //(1,1)
    Trot[6] = u.y*u.z*(1.f-cx)-u.x*sx; //(1,2)
    Trot[7] = 0.f; //(1,3)
    Trot[8] = u.z*u.x*(1.f-cx)-u.y*sx; //(2,0)
    Trot[9] = u.z*u.y*(1.f-cx)+u.x*sx;  //(2,1)
    Trot[10] = cx+u.z*u.z*(1.f-cx); //(2,2)
    Trot[11] = 0.f;//(2,3)

    //Multiply first two matrices and store in 'T'
    matmult(Trot,Ttrans,T);

    //Construct transformation matrix to translate back to 'origin'
    Ttrans[3] = origin.x; //(0,3)
    Ttrans[7] = origin.y; //(1,3)
    Ttrans[11] = origin.z;//(2,3)

    matmult(Ttrans,T,T);

}

void helios::makeTranslationMatrix( const helios::vec3& translation, float (&T)[16] ){

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

void helios::makeScaleMatrix( const helios::vec3& scale, float (&T)[16] ){

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

void helios::vecmult( const float M[16], const helios::vec3& v3, helios::vec3& result ){

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

float helios::deg2rad( float deg ){
  return deg*float(M_PI)/180.f;
}

float helios::rad2deg( float rad ){
  return rad*180.f/float(M_PI);
}

float helios::atan2_2pi(float y, float x){

  float v=0;

  if(x>0.f){
    v=atanf(y/x);
  }
  if(y>=0.f && x<0.f){
    v=float(M_PI)+atanf(y/x);
  }
  if(y<0.f && x<0.f){
    v=-float(M_PI)+atanf(y/x);
  }
  if(y>0.f && x==0.f){
    v=0.5f*float(M_PI);
  }
  if(y<0.f && x==0.f){
    v=-0.5f*float(M_PI);
  }
  if(v<0.f){
    v=v+2.f*float(M_PI);
  }
  return v;

}

SphericalCoord helios::cart2sphere( const vec3& Cartesian ){

  SphericalCoord Spherical;

  Spherical.radius = sqrt( Cartesian.x*Cartesian.x + Cartesian.y*Cartesian.y + Cartesian.z*Cartesian.z );

  Spherical.elevation = asin( Cartesian.z/Spherical.radius );

  Spherical.zenith = 0.5f*float(M_PI) - Spherical.elevation;

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
    if( !parse_float( tmp, o[c] ) ){
      o[c] = 99999;
    }
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
    if( !parse_float( tmp, o[c] ) ){
      o[c] = 99999;
    }
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
    if( !parse_float( tmp, o[c] ) ){
      o[c] = 99999;
    }
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
    if( !parse_int( tmp, o[c] ) ){
      o[c] = 99999;
    }
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
    if( !parse_int( tmp, o[c] ) ){
      o[c] = 99999;
    }
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
    if( !parse_int( tmp, o[c] ) ){
      o[c] = 99999;
    }
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
    if( !parse_float( tmp, o[c] ) ){
      o[c] = 0;
    }
    c++;
  }
  return make_RGBAcolor( o[0], o[1], o[2], o[3] );

}

bool helios::parse_float( const std::string &input_string, float &converted_float ){

  try {
    size_t read= 0;
    std::string str = trim_whitespace(input_string);
    double converted_double = std::stod(str, &read);
    converted_float = (float)converted_double;
    if (str.size() != read)
      return false;
  } catch ( std::invalid_argument& e ) {
    return false;
  }
  return true;

}

bool helios::parse_double( const std::string &input_string, double &converted_double ){

  try {
    size_t read= 0;
    std::string str = trim_whitespace(input_string);
    converted_double = std::stod(str, &read);
    if (str.size() != read)
      return false;
  } catch (std::invalid_argument& e) {
    return false;
  }
  return true;

}

bool helios::parse_int( const std::string &input_string, int &converted_int ){

  try {
    size_t read= 0;
    std::string str = trim_whitespace(input_string);
    converted_int = std::stoi(str, &read);
    if (str.size() != read)
      return false;
  } catch (std::invalid_argument& e) {
    return false;
  }
  return true;

}

bool helios::parse_uint( const std::string &input_string, uint &converted_uint ){

  try {
    size_t read= 0;
    std::string str = trim_whitespace(input_string);
    int converted_int = std::stoi(str, &read);
    if (str.size() != read || converted_int<0) {
      return false;
    }else{
      converted_uint = (uint)converted_int;
    }
  } catch (std::invalid_argument& e) {
    return false;
  }
  return true;

}


std::string helios::deblank(const char* input)
{
    int i,j;
    char output[255];
    std::strcpy(output,input);
    for (i = 0, j = 0; i<strlen(input); i++,j++)          
    {
        if (input[i]!=' ')                           
            output[j]=input[i];                     
        else
            j--;                                     
    }
    output[j]=0;
    std::string output_c = output;
    return output_c;

}

std::string helios::deblank(const std::string &input){
  return deblank(input.c_str());
}

std::string helios::trim_whitespace(const std::string &input){

  const std::string WHITESPACE = " \n\r\t\f\v";

  std::string outstring;

  size_t start = input.find_first_not_of(WHITESPACE);
  outstring = (start == std::string::npos) ? "" : input.substr(start);

  size_t end = outstring.find_last_not_of(WHITESPACE);
  outstring = (end == std::string::npos) ? "" : outstring.substr(0, end + 1);

  return outstring;

}

std::vector<std::string> helios::separate_string_by_delimiter( const std::string &inputstring, const std::string &delimiter ){

    std::vector<std::string> separated_string;

    if( inputstring.find(delimiter, 0)==std::string::npos ){
        return separated_string;
    }

    size_t p = 0;
    size_t p0 = 0;
    while ( p<inputstring.size()-1 ) {

        p = inputstring.find(delimiter, p0);
        separated_string.push_back( trim_whitespace(inputstring.substr(p0, p - p0)) );

        p0=p+1;
    }

    return separated_string;

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

template float helios::clamp<float>(float,float,float);
template double helios::clamp<double>(double,double,double);
template int helios::clamp<int>(int,int,int);
template uint helios::clamp<uint>(uint,uint,uint);

float helios::sum( const std::vector<float>& vect ){

    if( vect.empty() ){
        helios_runtime_error("ERROR (sum): Vector is empty.");
    }

    float m = 0;
    for( float i : vect){
        m += i;
    }

    return m;

}

float helios::mean( const std::vector<float>& vect ){

  if( vect.empty() ){
    helios_runtime_error("ERROR (mean): Vector is empty.");
  }

  float m = 0;
  for( float i : vect){
    m += i;
  }
  m /= float(vect.size());

  return m;

}

float helios::min( const std::vector<float>& vect ){

  if( vect.empty() ){
    helios_runtime_error("ERROR (min): Vector is empty.");
  }

  return *std::min_element(vect.begin(),vect.end());

}

int helios::min( const std::vector<int>& vect ){

  if( vect.empty() ){
      helios_runtime_error("ERROR (min): Vector is empty.");
  }

  return *std::min_element(vect.begin(),vect.end());

}

vec3 helios::min( const std::vector<vec3>& vect ){

  if( vect.empty() ){
      helios_runtime_error("ERROR (min): Vector is empty.");
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

float helios::max( const std::vector<float>& vect ){

  if( vect.empty() ){
      helios_runtime_error("ERROR (max): Vector is empty.");
  }

  return *std::max_element(vect.begin(),vect.end());

}

int helios::max( const std::vector<int>& vect ){

  if( vect.empty() ){
      helios_runtime_error("ERROR (max): Vector is empty.");
  }

  return *std::max_element(vect.begin(),vect.end());

}

vec3 helios::max( const std::vector<vec3>& vect ){

  if( vect.empty() ){
      helios_runtime_error("ERROR (max): Vector is empty.");
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

float helios::stdev( const std::vector<float> &vect ){

    if( vect.empty() ){
        helios_runtime_error("ERROR (stdev): Vector is empty.");
    }

    size_t size = vect.size();

    float m = 0;
    for(float i : vect){
        m += i;
    }
    m /= float(size);

    float stdev = 0;
    for(float i : vect){
        stdev += powf(i - m, 2.0);
    }

    return sqrtf(stdev/float(size));
}

float helios::median( std::vector<float> vect ){

    if( vect.empty() ){
        helios_runtime_error("ERROR (median): Vector is empty.");
    }

    size_t size = vect.size();

    sort(vect.begin(), vect.end());

    int mid = size/2;

    int midm1 = mid -1;

    float median ;
    if( size % 2 == 0)
    {
        median = (vect.at(mid) + vect.at(mid-1))/2.f;
    }else{
        median = vect.at(mid);
    }
    return median;
}

template <typename anytype>
void helios::resize_vector( std::vector<std::vector<anytype> > &vec, uint Nx, uint Ny ){
  vec.resize(Ny);
  for( int j=0; j<Ny; j++ ){
    vec.at(j).resize(Nx);
  }
}

template void helios::resize_vector<float>( std::vector<std::vector<float> > &, uint, uint );
template void helios::resize_vector<double>( std::vector<std::vector<double> > &, uint, uint );
template void helios::resize_vector<int>( std::vector<std::vector<int> > &, uint, uint );
template void helios::resize_vector<uint>( std::vector<std::vector<uint> > &, uint, uint );
template void helios::resize_vector<std::string>( std::vector<std::vector<std::string> > &, uint, uint );
template void helios::resize_vector<helios::vec2>( std::vector<std::vector<helios::vec2> > &, uint, uint );
template void helios::resize_vector<helios::vec3>( std::vector<std::vector<helios::vec3> > &, uint, uint );
template void helios::resize_vector<helios::vec4>( std::vector<std::vector<helios::vec4> > &, uint, uint );
template void helios::resize_vector<helios::int2>( std::vector<std::vector<helios::int2> > &, uint, uint );
template void helios::resize_vector<helios::int3>( std::vector<std::vector<helios::int3> > &, uint, uint );
template void helios::resize_vector<helios::int4>( std::vector<std::vector<helios::int4> > &, uint, uint );

template <typename anytype>
void helios::resize_vector( std::vector<std::vector<std::vector<anytype> > > &vec, uint Nx, uint Ny, uint Nz ){
  vec.resize(Nz);
  for( int k=0; k<Nz; k++ ){
    vec.at(k).resize(Ny);
    for( int j=0; j<Ny; j++ ){
      vec.at(k).at(j).resize(Nx);
    }
  }
}

template void helios::resize_vector<float>( std::vector<std::vector<std::vector<float> > > &, uint, uint, uint );
template void helios::resize_vector<double>( std::vector<std::vector<std::vector<double> > > &, uint, uint, uint );
template void helios::resize_vector<int>( std::vector<std::vector<std::vector<int> > > &, uint, uint, uint );
template void helios::resize_vector<uint>( std::vector<std::vector<std::vector<uint> > > &, uint, uint, uint );
template void helios::resize_vector<std::string>( std::vector<std::vector<std::vector<std::string> > > &, uint, uint, uint );
template void helios::resize_vector<helios::vec2>( std::vector<std::vector<std::vector<helios::vec2> > > &, uint, uint, uint );
template void helios::resize_vector<helios::vec3>( std::vector<std::vector<std::vector<helios::vec3> > > &, uint, uint, uint );
template void helios::resize_vector<helios::vec4>( std::vector<std::vector<std::vector<helios::vec4> > > &, uint, uint, uint );
template void helios::resize_vector<helios::int2>( std::vector<std::vector<std::vector<helios::int2> > > &, uint, uint, uint );
template void helios::resize_vector<helios::int3>( std::vector<std::vector<std::vector<helios::int3> > > &, uint, uint, uint );
template void helios::resize_vector<helios::int4>( std::vector<std::vector<std::vector<helios::int4> > > &, uint, uint, uint );

template <typename anytype>
void helios::resize_vector( std::vector<std::vector<std::vector<std::vector<anytype> > > > &vec, uint Nx, uint Ny, uint Nz, uint Nw ){
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

template void helios::resize_vector<float>( std::vector<std::vector<std::vector<std::vector<float> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<double>( std::vector<std::vector<std::vector<std::vector<double> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<int>( std::vector<std::vector<std::vector<std::vector<int> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<uint>( std::vector<std::vector<std::vector<std::vector<uint> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<std::string>( std::vector<std::vector<std::vector<std::vector<std::string> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<helios::vec2>( std::vector<std::vector<std::vector<std::vector<helios::vec2> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<helios::vec3>( std::vector<std::vector<std::vector<std::vector<helios::vec3> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<helios::vec4>( std::vector<std::vector<std::vector<std::vector<helios::vec4> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<helios::int2>( std::vector<std::vector<std::vector<std::vector<helios::int2> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<helios::int3>( std::vector<std::vector<std::vector<std::vector<helios::int3> > > > &, uint, uint, uint, uint );
template void helios::resize_vector<helios::int4>( std::vector<std::vector<std::vector<std::vector<helios::int4> > > > &, uint, uint, uint, uint );

Date helios::CalendarDay( int Julian_day, int year ){

  if( Julian_day<1 || Julian_day>366 ){
      helios_runtime_error("ERROR (CalendarDay): Julian day out of range.");
  }else if( year<1000 ){
      helios_runtime_error("ERROR (CalendarDay): Year should be specified in YYYY format.");
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
  month = 1;
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

int helios::JulianDay( const Date& date ){

  int day = date.day;
  int month = date.month;
  int year = date.year;

  if( day<1 || day>31 ){
      helios_runtime_error("ERROR (JulianDay): Day of month is out of range (day of " + std::to_string(day) + " was given).");
  }else if( month<1 || month>12){
      helios_runtime_error("ERROR (JulianDay): Month of year is out of range (month of " + std::to_string(month) + " was given).");
  }else if( year<1000 ){
      helios_runtime_error("ERROR (JulianDay): Year should be specified in YYYY format.");
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

    std::string fn = filename;
    if( fn.substr(fn.find_last_of(".") + 1) != "png" && fn.substr(fn.find_last_of(".") + 1) != "PNG" ) {
        helios_runtime_error("ERROR (readPNGAlpha): File " + fn + " is not PNG format.");
    }

  uint Nchannels;
  
  png_structp png_ptr;
  png_infop info_ptr;

  char header[8]; 

  /* open file and test for it being a png */
  FILE *fp = fopen(filename, "rb");
  if (!fp){
      helios_runtime_error("ERROR (PNGHasAlpha): File " + std::string(filename) + " could not be opened for reading. The file either does not exist or you do not have permission to read it.");
  }
  size_t result=fread(header, 1, 8, fp);
  
  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr){
      helios_runtime_error("ERROR (read_png_alpha): png_create_read_struct failed.");
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr){
      helios_runtime_error("ERROR (read_png_alpha): png_create_info_struct failed.");
  }
  
  if (setjmp(png_jmpbuf(png_ptr))){
      helios_runtime_error("ERROR (read_png_alpha): init_io failed.");
  }  

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  
  png_read_info(png_ptr, info_ptr);

  Nchannels = png_get_channels(png_ptr, info_ptr);

  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);

  if( Nchannels==4 ){
    return true;
  }else{
    return false;
  }

}

std::vector<std::vector<bool> > helios::readPNGAlpha( const std::string &filename ){

    std::string fn = filename;
    if( fn.substr(fn.find_last_of(".") + 1) != "png" && fn.substr(fn.find_last_of(".") + 1) != "PNG" ) {
        helios_runtime_error("ERROR (readPNGAlpha): File " + fn + " is not PNG format.");
    }

  int y;
  uint height, width;

  std::vector<std::vector<bool> > mask;
  
  png_structp png_ptr;
  png_infop info_ptr;
  png_bytep * row_pointers;

  char header[8];    // 8 is the maximum size that can be checked

  /* open file and test for it being a png */
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp){
      helios_runtime_error("ERROR (readPNGAlpha): File " + std::string(filename) + " could not be opened for reading.");
  }
  size_t result=fread(header, 1, 8, fp);
  // if (png_sig_cmp(header, 0, 8)){
  //   std::cerr << "ERROR (read_png_alpha): File " << filename << " is not recognized as a PNG file." << std::endl;
  //   exit(EXIT_FAILURE);
  // }

  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr){
      helios_runtime_error("ERROR (readPNGAlpha): png_create_read_struct failed.");
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr){
      helios_runtime_error("ERROR (readPNGAlpha): png_create_info_struct failed.");
  }
  
  if (setjmp(png_jmpbuf(png_ptr))){
      helios_runtime_error("ERROR (readPNGAlpha): init_io failed.");
  }  

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);
  
  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
//  color_type = png_get_color_type(png_ptr, info_ptr);
//  bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  mask.resize( height );
  for( uint i=0; i<height; i++ ){
    mask.at(i).resize(width);
  }

//  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  if (setjmp(png_jmpbuf(png_ptr))){
      helios_runtime_error("ERROR (read_png_alpha): read_image failed.");
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

  for(y = 0;y<height;y++)
    png_free (png_ptr,row_pointers[y]);
  png_free (png_ptr, row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);


  return mask;
  
}

void helios::readPNG( const std::string &filename, uint & width, uint & height, std::vector<helios::RGBAcolor> &texture ){

  std::string fn = filename;
  if( fn.substr(fn.find_last_of(".") + 1) != "png" && fn.substr(fn.find_last_of(".") + 1) != "PNG" ){
    helios_runtime_error("ERROR (readPNG): File " + fn + " is not PNG format.");
  }

  int x, y;

  png_byte color_type;
  png_byte bit_depth;

  png_structp png_ptr;
  png_infop info_ptr;
  int number_of_passes;
  png_bytep * row_pointers;

  char header[8];    // 8 is the maximum size that can be checked

  /* open file and test for it being a png */
  FILE *fp = fopen(filename.c_str(), "rb");
  if (!fp){
    helios_runtime_error("ERROR (readPNG): File " + filename + "could not be opened for reading.");
   }
   size_t result=fread(header, 1, 8, fp);

  /* initialize stuff */
  png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);

  if (!png_ptr){
    helios_runtime_error("ERROR (readPNG): failed to create PNG read structure.");
  }

  info_ptr = png_create_info_struct(png_ptr);
  if (!info_ptr){
    helios_runtime_error("ERROR (readPNG): failed to create PNG inof structure.");
  }

  if (setjmp(png_jmpbuf(png_ptr))){
    helios_runtime_error("ERROR (readPNG): init_io failed.");
  }

  png_init_io(png_ptr, fp);
  png_set_sig_bytes(png_ptr, 8);

  png_read_info(png_ptr, info_ptr);

  width = png_get_image_width(png_ptr, info_ptr);
  height = png_get_image_height(png_ptr, info_ptr);
  color_type = png_get_color_type(png_ptr, info_ptr);
  bit_depth = png_get_bit_depth(png_ptr, info_ptr);

  number_of_passes = png_set_interlace_handling(png_ptr);
  png_read_update_info(png_ptr, info_ptr);

  /* read file */
  if (setjmp(png_jmpbuf(png_ptr))){
    helios_runtime_error("ERROR (readPNG): PNG read failed.");
  }

  row_pointers = (png_bytep*) malloc(sizeof(png_bytep) * height);
  for (y=0; y<height; y++)
    row_pointers[y] = (png_byte*) malloc(png_get_rowbytes(png_ptr,info_ptr));

  png_read_image(png_ptr, row_pointers);

  fclose(fp);

  texture.resize(height*width);

  for (uint j=0; j<height; j++){
    png_byte* row=row_pointers[j];
    for (int i=0; i < width; i++ ){
      png_byte* ba=&row[i*4];
      texture.at(j*width+i).r = (float)ba[0]/255.f;
      texture.at(j*width+i).g = (float)ba[1]/255.f;
      texture.at(j*width+i).b = (float)ba[2]/255.f;
      texture.at(j*width+i).a = (float)ba[3]/255.f;
    }
  }

  for(y = 0;y<height;y++)
    png_free (png_ptr,row_pointers[y]);
  png_free (png_ptr, row_pointers);
  png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);


}

void helios::writePNG( const std::string &filename, uint width, uint height, const std::vector<helios::RGBAcolor> &pixel_data ) {
  int y;

  FILE *fp = fopen(filename.c_str(), "wb");
  if(!fp){
    helios_runtime_error("ERROR (writePNG): failed to open image file.");
  }

  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png){
    helios_runtime_error("ERROR (writePNG): failed to create PNG write structure.");
  }

  png_infop info = png_create_info_struct(png);
  if (!info){
    helios_runtime_error("ERROR (writePNG): failed to create PNG info structure.");
  }

  if (setjmp(png_jmpbuf(png))){
    helios_runtime_error("ERROR (writePNG): init_io failed.");
  }

  png_init_io(png, fp);

  // Output is 8bit depth, RGBA format.
  png_set_IHDR(
      png,
      info,
      width,
      height,
      8,
      PNG_COLOR_TYPE_RGBA,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT,
      PNG_FILTER_TYPE_DEFAULT
  );
  png_write_info(png, info);

  // To remove the alpha channel for PNG_COLOR_TYPE_RGB format,
  // Use png_set_filler().
  //png_set_filler(png, 0, PNG_FILLER_AFTER);

  std::vector<unsigned char*> row_pointers;
  row_pointers.resize(height);

  std::vector<std::vector<unsigned char> > data;
  data.resize(height);

  for( uint row=0; row<height; row++ ) {
    data.at(row).resize(4*width);
    for( uint col=0; col<width; col++ ){
      data.at(row).at(4*col) = (unsigned char)round(clamp(pixel_data.at(row*width+col).r,0.f,1.f)*255.f);
      data.at(row).at(4*col+1) = (unsigned char)round(clamp(pixel_data.at(row*width+col).g,0.f,1.f)*255.f);
      data.at(row).at(4*col+2) = (unsigned char)round(clamp(pixel_data.at(row*width+col).b,0.f,1.f)*255.f);
      data.at(row).at(4*col+3) = (unsigned char)round(clamp(pixel_data.at(row*width+col).a,0.f,1.f)*255.f);
    }
    row_pointers.at(row) = &data.at(row).at(0);
  }

  png_write_image(png, &row_pointers.at(0) );
  png_write_end(png, nullptr);

  fclose(fp);

  png_destroy_write_struct(&png, &info);
}

void helios::readJPEG( const std::string &filename, uint &width, uint &height, std::vector<helios::RGBcolor> &pixel_data ) {

    auto file_extension = getFileExtension(filename);
    if ( file_extension != ".jpg" && file_extension != ".JPG" && file_extension != ".jpeg" && file_extension != ".JPEG" ) {
        throw (std::runtime_error("ERROR (Context::readJPEG): File " + filename + " is not JPEG format."));
    }

    struct jpeg_decompress_struct cinfo;

    FILE * infile;		/* source file */
    JSAMPARRAY buffer;		/*output row buffer */
    int row_stride;

    if ((infile = fopen(filename.c_str(), "rb")) == nullptr ) {
        throw (std::runtime_error("ERROR (Context::readJPEG): File " + filename + " could not be opened. Check that the file exists and that you have permission to read it."));
    }

    jpeg_create_decompress(&cinfo);

    jpeg_stdio_src(&cinfo, infile);

    (void) jpeg_read_header(&cinfo, TRUE);

    (void) jpeg_start_decompress(&cinfo);

    row_stride = cinfo.output_width * cinfo.output_components;
    buffer = (*cinfo.mem->alloc_sarray) ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, 1);

    width=cinfo.output_width;
    height=cinfo.output_height;

    if(cinfo.output_components!=3){
        throw (std::runtime_error("ERROR (Context::readJPEG): Image file does not have RGB components."));
    }

    pixel_data.resize(width*height);

    JSAMPLE* ba;
    int row = 0;
    while (cinfo.output_scanline < cinfo.output_height) {
        (void) jpeg_read_scanlines(&cinfo, buffer, 1);

        ba=buffer[0];

        for (int col=0; col < row_stride; col+=3){
            pixel_data.at(row*width+col) = make_RGBcolor(ba[row]/255.f,ba[row+1]/255.f,ba[row+2]/255.f);
        }

        row++;
    }

    (void) jpeg_finish_decompress(&cinfo);

    jpeg_destroy_decompress(&cinfo);

    fclose(infile);

}

void helios::writeJPEG( const std::string &a_filename, uint width, uint height, const std::vector<helios::RGBcolor> &pixel_data ) {

    struct my_error_mgr {

        struct jpeg_error_mgr pub;	/* "public" fields */

        jmp_buf setjmp_buffer;	/* for return to caller */
    };

    std::string filename = a_filename;
    auto file_extension = getFileExtension(filename);
    if ( file_extension != ".jpg" && file_extension != ".JPG" && file_extension != ".jpeg" && file_extension != ".JPEG" ) {
        filename.append(".jpeg");
    }

    if( pixel_data.size()!=width*height ){
        helios_runtime_error("ERROR (Context::writeJPEG): Pixel data does not have size of width*height.");
    }

    const uint bsize = 3 * width * height;
    std::vector<unsigned char> screen_shot_trans(bsize);

    size_t ii = 0;
    for( size_t i=0; i<width*height; i++ ){
        screen_shot_trans.at(ii) = (unsigned char)round(clamp(pixel_data.at(i).r,0.f,1.f)*255);
        screen_shot_trans.at(ii+1) = (unsigned char)round(clamp(pixel_data.at(i).g,0.f,1.f)*255);
        screen_shot_trans.at(ii+2) = (unsigned char)round(clamp(pixel_data.at(i).b,0.f,1.f)*255);
        ii+=3;
    }

    struct jpeg_compress_struct cinfo;

    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);

    /* More stuff */
    FILE * outfile;		/* target file */
    JSAMPROW row_pointer;	/* pointer to JSAMPLE row[s] */
    int row_stride;

    /* Now we can initialize the JPEG compression object. */
    jpeg_create_compress(&cinfo);

    if ((outfile = fopen(filename.c_str(), "wb")) == nullptr ) {
        helios_runtime_error("ERROR (Context::writeJPEG): File " + filename + " could not be opened. Check that the file path is correct you have permission to write to it.");
    }
    jpeg_stdio_dest(&cinfo, outfile);

    cinfo.image_width = width; 	/* image width and height, in pixels */
    cinfo.image_height = height;
    cinfo.input_components = 3;		/* # of color components per pixel */
    cinfo.in_color_space = JCS_RGB; 	/* colorspace of input image */

    jpeg_set_defaults(&cinfo);

    jpeg_set_quality(&cinfo, 100, TRUE /* limit to baseline-JPEG values */);

    jpeg_start_compress(&cinfo, TRUE);

    row_stride = width * 3;	/* JSAMPLEs per row in image_buffer */

    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer = (JSAMPROW) &screen_shot_trans[ (cinfo.image_height-cinfo.next_scanline-1) * row_stride ];
        (void) jpeg_write_scanlines(&cinfo, &row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    /* After finish_compress, we can close the output file. */
    fclose(outfile);

    jpeg_destroy_compress(&cinfo);

}

std::vector<int> helios::flatten( const std::vector<std::vector<int> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<int> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<uint> helios::flatten( const std::vector<std::vector<uint> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<uint> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<float> helios::flatten( const std::vector<std::vector<float> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<float> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<double> helios::flatten( const std::vector<std::vector<double> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<double> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<helios::vec2> helios::flatten( const std::vector<std::vector<helios::vec2> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<helios::vec2> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<helios::vec3> helios::flatten( const std::vector<std::vector<helios::vec3> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<helios::vec3> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<helios::vec4> helios::flatten( const std::vector<std::vector<helios::vec4> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<helios::vec4> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<helios::int2> helios::flatten( const std::vector<std::vector<helios::int2> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<helios::int2> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<helios::int3> helios::flatten( const std::vector<std::vector<helios::int3> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<helios::int3> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<helios::int4> helios::flatten( const std::vector<std::vector<helios::int4> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<helios::int4> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<std::string> helios::flatten( const std::vector<std::vector<std::string> > &vec ){

    size_t ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            ind++;
        }
    }

    std::vector<std::string> flat( ind );
    ind = 0;
    for( const auto &j : vec ) {
        for( const auto& i : j ) {
            flat[ind] = i;
            ind++;
        }
    }
    return flat;

}

std::vector<int> helios::flatten( const std::vector<std::vector<std::vector<int> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<int> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<uint> helios::flatten( const std::vector<std::vector<std::vector<uint> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<uint> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<float> helios::flatten( const std::vector<std::vector<std::vector<float> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<float> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<double> helios::flatten( const std::vector<std::vector<std::vector<double> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<double> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<helios::vec2> helios::flatten( const std::vector<std::vector<std::vector<helios::vec2> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<helios::vec2> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<helios::vec3> helios::flatten( const std::vector<std::vector<std::vector<helios::vec3> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<helios::vec3> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<helios::vec4> helios::flatten( const std::vector<std::vector<std::vector<helios::vec4> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<helios::vec4> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<helios::int2> helios::flatten( const std::vector<std::vector<std::vector<helios::int2> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<helios::int2> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<helios::int3> helios::flatten( const std::vector<std::vector<std::vector<helios::int3> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<helios::int3> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<helios::int4> helios::flatten( const std::vector<std::vector<std::vector<helios::int4> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<helios::int4> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<std::string> helios::flatten( const std::vector<std::vector<std::vector<std::string> > > &vec ){

    size_t ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                ind++;
            }
        }
    }

    std::vector<std::string> flat( ind );
    ind = 0;
    for( const auto &k : vec ) {
        for( const auto &j : k ) {
            for( const auto& i : j ) {
                flat[ind] = i;
                ind++;
            }
        }
    }
    return flat;

}

std::vector<int> helios::flatten( const std::vector<std::vector<std::vector<std::vector<int> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<int> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<uint> helios::flatten( const std::vector<std::vector<std::vector<std::vector<uint> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<uint> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<float> helios::flatten( const std::vector<std::vector<std::vector<std::vector<float> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<float> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<double> helios::flatten( const std::vector<std::vector<std::vector<std::vector<double> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<double> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<helios::vec2> helios::flatten( const std::vector<std::vector<std::vector<std::vector<helios::vec2> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<helios::vec2> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<helios::vec3> helios::flatten( const std::vector<std::vector<std::vector<std::vector<helios::vec3> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<helios::vec3> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<helios::vec4> helios::flatten( const std::vector<std::vector<std::vector<std::vector<helios::vec4> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<helios::vec4> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<helios::int2> helios::flatten( const std::vector<std::vector<std::vector<std::vector<helios::int2> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<helios::int2> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<helios::int3> helios::flatten( const std::vector<std::vector<std::vector<std::vector<helios::int3> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<helios::int3> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<helios::int4> helios::flatten( const std::vector<std::vector<std::vector<std::vector<helios::int4> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<helios::int4> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

std::vector<std::string> helios::flatten( const std::vector<std::vector<std::vector<std::vector<std::string> > > > &vec ){

    size_t ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    ind++;
                }
            }
        }
    }

    std::vector<std::string> flat( ind );
    ind = 0;
    for( const auto &w : vec ) {
        for( const auto &k : w) {
            for (const auto &j: k) {
                for (const auto &i: j) {
                    flat[ind] = i;
                    ind++;
                }
            }
        }
    }
    return flat;

}

helios::vec3 helios::spline_interp3(float u, const vec3 &x_start, const vec3 &tan_start, const vec3 &x_end, const vec3 &tan_end ){

  //Perform interpolation between two 3D points using Cubic Hermite Spline

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
    if( !parse_float( field_str, value ) ){
      value = 99999;
    }
  }
  
  return value;
  
}

int helios::XMLloadint( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  int value;
  if( strlen(field_str)==0 ){
    value = 99999;
  }else{
    if( !parse_int( field_str, value ) ){
      value = 99999;
    }
  }
  
  return value;
  
}

std::string helios::XMLloadstring( pugi::xml_node node, const char* field ){

  std::string field_str = deblank(node.child_value(field));
    
  std::string value;
  if( field_str.empty() ){
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
    value = make_RGBAcolor(1,1,1,0);
  }else{
    value = string2RGBcolor( field_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
  }
  
  return make_RGBcolor(value.r,value.g,value.b);
  
}

helios::RGBAcolor helios::XMLloadrgba( pugi::xml_node node, const char* field ){

  const char* field_str = node.child_value(field);
    
  helios::RGBAcolor value;
  if( strlen(field_str)==0 ){
    value = make_RGBAcolor(1,1,1,0);
  }else{
    value = string2RGBcolor( field_str ); //note: pugi loads xml data as a character.  need to separate it into 3 floats
  }
  
  return value;
  
}

float helios::fzero(float(*function)(float value, std::vector<float> &variables, const void *parameters), std::vector<float> &variables, const void *parameters, float init_guess, float err_tol, int max_iterations ){

    float T;

    float T_old_old = 1.1f*init_guess;

    float T_old = init_guess;

    float resid_old = function(T_old,variables,parameters);
    float resid_old_old = function(T_old_old,variables,parameters);

    float resid = 100;
    float err = resid;
    uint iter = 0;
    while( err>err_tol && iter<max_iterations ){

        if( resid_old==resid_old_old ){//this condition will cause NaN
            err=0;
            break;
        }

        T = fabs((T_old_old*resid_old-T_old*resid_old_old)/(resid_old-resid_old_old));

        resid = function(T,variables,parameters);

        resid_old_old = resid_old;
        resid_old = resid;

        err = fabs(T_old-T_old_old)/fabs(T_old_old);

        T_old_old = T_old;
        T_old = T;

        iter++;

    }

    if( err>err_tol ){
        printf("WARNING: fzero solution did not converge.\n");
    }

    return T;

}

float helios::interp1( const std::vector<helios::vec2> &points, float x ) {

    //Ensure that no 2 adjacent x values are equal, and that x values are monotonically increasing
    const float EPSILON{1.0E-5};
    for (std::size_t i = 1; i < points.size(); ++i) {
        float deltaX{std::abs(points[i].x - points[i - 1].x)};
        if (deltaX < EPSILON) {
            if (deltaX > -EPSILON) {
                helios_runtime_error("ERROR (interp1): Adjacent X points cannot be equal.");
            } else {
                helios_runtime_error("ERROR (interp1): X points must increase monotonically.");
            }
        }
    }

    //Define a lambda that returns true if the x value
    //of a point pair is < the caller's x value
    auto lessThan = [](const vec2 &point, float x) {
        return point.x < x;
    };

    //Find the first table entry whose value is >= caller's x value
    auto iter = std::lower_bound(points.cbegin(), points.cend(), x, lessThan);

    //If the caller's X value is greater than the largest
    //X value in the table, we can't interpolate.
    if (iter == points.cend()) {
        return (points.cend() - 1)->y;
    }

    //If the caller's X value is less than the smallest X value in the table,
    //we can't interpolate.
    if (iter == points.cbegin() and x <= points.cbegin()->x) {
        return points.cbegin()->y;
    }

    //We can interpolate!
    float upperX{iter->x};
    float upperY{iter->y};
    float lowerX{(iter - 1)->x};
    float lowerY{(iter - 1)->y};

    float deltaY{upperY - lowerY};
    float deltaX{upperX - lowerX};

    return lowerY + ((x - lowerX)/ deltaX) * deltaY;

}

float helios::point_distance( const helios::vec3 &p1 , const helios::vec3 &p2){
    return (p1-p2).magnitude();
}

std::string helios::getFileExtension( const std::string &filepath ){
  std::string ext;

  if( filepath.find_last_of('.')<filepath.size() ){
    ext = filepath.substr(filepath.find_last_of('.'));
  }else { // does not contain any .'s
    return "";
  }

  //edge case when file is in a hidden directory AND there is no file extension (return empty string)
  if( filepath.find_last_of('/')<filepath.size() && filepath.at(filepath.find_last_of('.')-1)=='/' ){
    ext = "";
  }

  //edge case when file is in a hidden directory AND there is no file extension AND file path starts with '.' (return empty string)
  if( filepath.find_last_of('.')==0 ){
    ext = "";
  }

  return ext;
}

std::string helios::getFileStem( const std::string &filepath ){
  std::string fn = filepath;
  if( filepath.find('/')<filepath.size() ) {
    fn = filepath.substr(filepath.find_last_of('/') + 1);
  }else if( filepath.find('\\')<filepath.size() ){
    fn = filepath.substr(filepath.find_last_of('\\') + 1);
  }

  if( fn.find('.')<fn.size() ) {
    fn = fn.substr( 0, fn.find_last_of('.') );
  }

  return fn;
}

std::string helios::getFileName( const std::string &filepath ){
  if( filepath.find('/')<filepath.size() ) {
    return filepath.substr(filepath.find_last_of('/') + 1);
  }else if( filepath.find('\\')==filepath.size() ) {
    return filepath.substr(filepath.find_last_of('\\') + 1);
  }else{
    return filepath;
  }

}

std::string helios::getFilePath( const std::string &filepath, bool trailingslash ){
  if( filepath.find('/')==filepath.size() ){
    if( trailingslash ){
      std::string str = "/";
      return str;
    }else {
      std::string str;
      return str;
    }
  }else{
    if( trailingslash ) {
      return filepath.substr(0, filepath.find_last_of('/') + 1);
    }else{
      return filepath.substr(0, filepath.find_last_of('/'));
    }
  }
}

std::vector<float> helios::importVectorFromFile(const std::string &filepath){

  std::ifstream stream(filepath.c_str());

  if( !stream.is_open() ){
    helios_runtime_error("ERROR (helios::importVectorFromFile): File " + filepath + " could not be opened for reading. Check that it exists and that you have permission to read it.");
  }

  std::istream_iterator<float> start(stream), end;
  std::vector<float> vec(start, end);
  return vec;

}
