/** \file "AerialLiDAR.cpp" Source file for LiDAR plug-in dealing with aerial scans.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "AerialLiDAR.h"

using namespace std;
using namespace helios;

AerialScanMetadata::AerialScanMetadata( const helios::vec3 __center, const helios::vec2 __extent, const float __coneangle, const float __scandensity, const float __exitDiameter, const float __beamDivergence ){

  //Copy arguments into structure variables
  center = __center;
  extent = __extent;
  coneangle = __coneangle;
  scandensity = __scandensity;
  exitDiameter = __exitDiameter;
  beamDivergence = __beamDivergence;

}
 
AerialLiDARcloud::AerialLiDARcloud( void ){

  hitgridcellcomputed = false;
  groundheightcomputed = false;
  printmessages = true;
  
}

AerialLiDARcloud::~AerialLiDARcloud( void ){
  
}

int AerialLiDARcloud::selfTest(void){

  float err_tol = 0.1;

  int fail_count = 0;

  //------- dense vegatation test to check "mean dr" method for calculating LAD -------//

  std::cout << "Running aerial LiDAR dense vegetation test..." << std::flush;

  Context context_1;

  vec3 boxsize(10,10,10);

  int3 Nleaves(100,100,100);

  float L = 0.05;

  bool flag;
  float LAD_exact;

  for( int k=0; k<Nleaves.z; k++ ){
    for( int j=0; j<Nleaves.y; j++ ){
      for( int i=0; i<Nleaves.x; i++ ){

  	vec3 x( context_1.randu()*boxsize.x, context_1.randu()*boxsize.y, context_1.randu()*boxsize.z );

  	float theta = acos(1.f-context_1.randu());
  	float phi = context_1.randu()*2.f*M_PI;

  	context_1.addPatch( x, make_vec2(L,L), make_SphericalCoord(theta,phi) );
	
      }
    }
  }

  context_1.addPatch( make_vec3(0.5*boxsize.x,0.5*boxsize.y,-0.001), make_vec2( boxsize.x, boxsize.y ) );

  LAD_exact = float(Nleaves.x*Nleaves.y*Nleaves.z)*L*L/(boxsize.x*boxsize.y*boxsize.z);

  flag = true;
  AerialLiDARcloud lidar_1;
  lidar_1.disableMessages();

  lidar_1.syntheticScan( &context_1, "plugins/aeriallidar/xml/synthetic_aerial_test.xml" );
  
  helios::vec3 center=lidar_1.getGridCenter();
  if( center.x!=5.0 || center.y!=5.0 || center.z!=5.0 ){
    std::cout << "Grid center was not set correctly." << std::endl;
    flag = false;
  }

  helios::int3 resolution=lidar_1.getGridResolution();
  if( resolution.x!=2 || resolution.y!=2 || resolution.z!=2 ){
    std::cout << "Grid resolution was not set correctly." << std::endl;
    flag = false;
  }

  int v = 6;
  helios::int3 ijk = lidar_1.gridindex2ijk(v);
  if( v!=lidar_1.gridijk2index(ijk) ){
    printf("ijk = (%d,%d,%d) %d\n",ijk.x,ijk.y,ijk.z,lidar_1.gridijk2index(ijk));
    flag = false;
    std::cout << "ijk failed" << std::endl;
  }

  lidar_1.calculateLeafAreaGPU( 0.5, 10 );

  for( int v=4; v<8; v++ ){

    float LAD = lidar_1.getCellLeafAreaDensity( lidar_1.gridindex2ijk(v) );

    if( fabs(LAD-LAD_exact)/LAD_exact > err_tol ){
      flag = false;
    }

  }

  if( flag ){
    std::cout << "passed." << std::endl;
  }else{
    std::cout << "failed." << std::endl;
    fail_count ++;
  }

  //------- sparse vegatation test to check "mean P" method for calculating LAD -------//

  std::cout << "Running aerial LiDAR sparse vegetation test..." << std::flush;

  Context context_2;

  Nleaves = make_int3(25,25,25);

  for( int k=0; k<Nleaves.z; k++ ){
    for( int j=0; j<Nleaves.y; j++ ){
      for( int i=0; i<Nleaves.x; i++ ){

  	vec3 x( context_2.randu()*boxsize.x, context_2.randu()*boxsize.y, context_2.randu()*boxsize.z );

  	float theta = acos(1.f-context_2.randu());
  	float phi = context_2.randu()*2.f*M_PI;

  	context_2.addPatch( x, make_vec2(L,L), make_SphericalCoord(theta,phi) );
	
      }
    }
  }

  context_2.addPatch( make_vec3(0.5*boxsize.x,0.5*boxsize.y,-0.001), make_vec2( boxsize.x, boxsize.y ) );

  LAD_exact = float(Nleaves.x*Nleaves.y*Nleaves.z)*L*L/(boxsize.x*boxsize.y*boxsize.z);

  AerialLiDARcloud lidar_2;
  lidar_2.disableMessages();

  lidar_2.syntheticScan( &context_2, "plugins/aeriallidar/xml/synthetic_aerial_test.xml" );

  lidar_2.calculateLeafAreaGPU( 0.5, 10 );

  flag = true;
  for( int v=0; v<8; v++ ){

    float LAD = lidar_2.getCellLeafAreaDensity( lidar_2.gridindex2ijk(v) );

    if( fabs(LAD-LAD_exact)/LAD_exact > 1.5*err_tol ){
      flag = false;
    }

  }

  if( flag ){
    std::cout << "passed." << std::endl;
  }else{
    std::cout << "failed." << std::endl;
    fail_count ++;
  }

  //------- sparse vegatation ground and canopy height estimation -------//

  std::cout << "Running aerial LiDAR ground and canopy height test..." << std::flush;

  Context context_3;

  Nleaves = make_int3(25,25,35);

  for( int k=0; k<Nleaves.z; k++ ){
    for( int j=0; j<Nleaves.y; j++ ){
      for( int i=0; i<Nleaves.x; i++ ){

	vec3 x( context_3.randu()*boxsize.x, context_3.randu()*boxsize.y, context_3.randu()*boxsize.z );

	float theta = acos(1.f-context_3.randu());
	float phi = context_3.randu()*2.f*M_PI;

	context_3.addPatch( x, make_vec2(L,L), make_SphericalCoord(theta,phi) );
	
      }
    }
  }

  float zground = 0.2;

  context_3.addPatch( make_vec3(0.5*boxsize.x,0.5*boxsize.y,zground), make_vec2( boxsize.x, boxsize.y ) );

  AerialLiDARcloud lidar_3;
  lidar_3.disableMessages();

  lidar_3.syntheticScan( &context_3, "plugins/aeriallidar/xml/synthetic_aerial_test.xml" );

  for( int r=0; r<lidar_3.getHitCount(); r++ ){

    lidar_3.setHitData( r, "target_index", 1 );
    lidar_3.setHitData( r, "target_count", 1 ); 
        
  }
  
  lidar_3.generateHeightModel( 100, 0.5, 0.1, 0.5, 0.1 );

  flag = true;
  for( int v=0; v<8; v++ ){

    int3 index = lidar_3.gridindex2ijk(v);
    float zg = lidar_3.getCellGroundHeight( make_int2(index.x,index.y) );

    if( fabs(zg-zground)/fabs(zground) > 1.5*err_tol ){
      flag = false;
    }

  }

  for( int r=0; r<lidar_3.getHitCount(); r++ ){

    vec3 xyz = lidar_3.getHitXYZ(r);

    if( fabs(xyz.z-zground)>9 ){
      lidar_3.setHitData( r, "target_index", 1 );
    }else{
      lidar_3.setHitData( r, "target_index", 2 );
    }
    lidar_3.setHitData( r, "target_count", 2 ); 
    
  }

  lidar_3.generateHeightModel( 400, 0.5, 0.1, 1.0, 0.2 );

  for( int v=0; v<8; v++ ){

    int3 index = lidar_3.gridindex2ijk(v);
    float zc = lidar_3.getCellVegetationHeight( make_int2(index.x,index.y) );
    float zm = lidar_3.getCellMaximumHitHeight( make_int2(index.x,index.y) );

    if( fabs(zc-(boxsize.z-0.5))/fabs(boxsize.z-0.5) > 1.5*err_tol ){
      flag = false;
    }else if( fabs(zm-boxsize.z)/fabs(boxsize.z) > err_tol ){
      flag = false;
    }
      
  }

  if( flag ){
    std::cout << "passed." << std::endl;
  }else{
    std::cout << "failed." << std::endl;
    fail_count ++;
  }


  if( fail_count==0 ){
    return 0;
  }else{
    return 1;
  }
    
}

void AerialLiDARcloud::disableMessages( void ){
  printmessages = false;
}

void AerialLiDARcloud::enableMessages( void ){
  printmessages = true;
}

uint AerialLiDARcloud::getScanCount( void ){
  return scans.size();
}

void AerialLiDARcloud::addScan( const AerialScanMetadata newscan ){
  scans.push_back(newscan);
}

void AerialLiDARcloud::addHitPoint( const uint scanID, const helios::vec3 hit_xyz, helios::vec3 ray_origin ){

  vec3 direction_cart = hit_xyz-ray_origin;
  SphericalCoord direction_sphere = cart2sphere(direction_cart);

  //default color
  RGBcolor color = make_RGBcolor(1,0,0);

  //empty data
  std::map<std::string, float> data;
  
  addHitPoint( scanID, hit_xyz, direction_sphere, color, data );

}

void AerialLiDARcloud::addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction ){

  //default color
  RGBcolor color = make_RGBcolor(1,0,0);

  //empty data
  std::map<std::string, float> data;
  
  addHitPoint( scanID, hit_xyz, direction, color, data );

}

void AerialLiDARcloud::addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction, const std::map<std::string, float> data ){

  //default color
  RGBcolor color = make_RGBcolor(1,0,0);

  addHitPoint( scanID, hit_xyz, direction, color, data );
  
}

void AerialLiDARcloud::addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction, const helios::RGBcolor color ){

  //empty data
  std::map<std::string, float> data;
  
  addHitPoint( scanID, hit_xyz, direction, color, data );
  
}

void AerialLiDARcloud::addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction, const helios::RGBcolor color, const std::map<std::string, float> data ){

  //error checking
  if( scanID>=scans.size() ){
    helios_runtime_error("ERROR (AerialLiDARcloud::addHitPoint): Hit point cannot be added to scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }

  AerialScanMetadata scan = scans.at(scanID);

  AerialHitPoint hit( scanID, hit_xyz, direction, make_int2(0,0), color, data ); 

  hits.push_back( hit );

}

void AerialLiDARcloud::deleteHitPoint( const uint index ){

  if( index>=hits.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::deleteHitPoint): Hit point #" + std::to_string(index) + " cannot be deleted from the scan because there have only been " + std::to_string(hits.size()) +" hit points added.");
  }
  
  AerialHitPoint hit = hits.at(index);

  int scanID = hit.scanID;

  //erase from vector of hits (use swap-and-pop method)
  std::swap( hits.at(index), hits.back() );
  hits.pop_back();
  
}

uint AerialLiDARcloud::getHitCount( void ) const{
  return hits.size();
}

helios::vec3 AerialLiDARcloud::getScanCenter( const uint scanID ) const{
  if( scanID>=scans.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getScanCenter): Cannot get center of scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }
  return scans.at(scanID).center;
}

helios::vec2 AerialLiDARcloud::getScanExtent( const uint scanID ) const{
  if( scanID>=scans.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getScanExtent): Cannot get extent for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }
  return scans.at(scanID).extent;
}

float AerialLiDARcloud::getScanConeAngle( const uint scanID ) const{
  if( scanID>=scans.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getScanConeAngle): Cannot get cone angle for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }
  return scans.at(scanID).coneangle;
}

float AerialLiDARcloud::getScanDensity( const uint scanID ) const{
  if( scanID>=scans.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getScanDensity): Cannot get point density for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }
  return scans.at(scanID).scandensity;
}

float AerialLiDARcloud::getScanBeamExitDiameter( const uint scanID ) const{
  if( scanID>=scans.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getScanBeamExitDiameter): Cannot get exit diameter for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }
  return scans.at(scanID).exitDiameter;
}

float AerialLiDARcloud::getScanBeamDivergence( const uint scanID ) const{
  if( scanID>=scans.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getScanBeamDivergence): Cannot get beam divergence for scan #" + std::to_string(scanID) + " because there have only been " + std::to_string(scans.size()) + " scans added.");
  }
  return scans.at(scanID).beamDivergence;
}

helios::vec3 AerialLiDARcloud::getHitXYZ( const uint index ) const{

  if( index>=hits.size() ){
      helios_runtime_error("ERROR (AerialLiDARcloud::getHitXYZ): Hit point index out of bounds. Requesting hit #" + std::to_string(index) + " but scan only has " + std::to_string(hits.size()) + " hits.");
  }

  return hits.at(index).position;

}

helios::SphericalCoord AerialLiDARcloud::getHitRaydir( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitRaydir): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  return hits.at(index).direction;

}

float AerialLiDARcloud::getHitData( const uint index, const char* label ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitData): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  std::map<std::string, float> hit_data = hits.at(index).data;
  if( hit_data.find(label) == hit_data.end() ){
    cerr << "ERROR (getHitData): Data value ``" << label << "'' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  return hit_data.at(label);

}

void AerialLiDARcloud::setHitData( const uint index, const char* label, const float value ){

  if( index>=hits.size() ){
    cerr << "ERROR (setHitScalarData): Hit point index out of bounds. Tried to set hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  hits.at(index).data[label] = value;
  
}

bool AerialLiDARcloud::doesHitDataExist( const uint index, const char* label ) const{

  if( index>=hits.size() ){
    return false;
  }

  std::map<std::string, float> hit_data = hits.at(index).data;
  if( hit_data.find(label) == hit_data.end() ){
    return false;
  }else{
    return true;
  }

}

RGBcolor AerialLiDARcloud::getHitColor( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitColor): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  return hits.at(index).color;

}

int AerialLiDARcloud::getHitScanID( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitColor): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  return hits.at(index).scanID;
  
}

helios::int3 AerialLiDARcloud::getHitGridCell( const uint index ) const{
  
  if( index>=hits.size() ){
    cerr << "ERROR (getHitGridCell): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }else if( !hitgridcellcomputed ){
    cerr << "WARNING (getHitGridCell): calculateHitGridCell() must be run prior to calling this function" << std::endl;
    return make_int3(-1,-1,-1);
  }

  return hits.at(index).gridcell;
  
}

void AerialLiDARcloud::setHitGridCell( const uint index, const helios::int3 ijk ){

  if( index>=hits.size() ){
    cerr << "ERROR (setHitGridCell): Hit point index out of bounds. Tried to set hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.x>=gridresolution.x ){
    cerr << "ERROR (setHitGridCell): Grid index in x-direction was specified as " << ijk.x << ", which is larger than the maximum value based on the grid dimension of " << gridresolution.x-1 << "." << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y>=gridresolution.y ){
    cerr << "ERROR (setHitGridCell): Grid index in y-direction was specified as " << ijk.y << ", which is larger than the maximum value based on the grid dimension of " << gridresolution.y-1 << "." << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z>=gridresolution.z ){
    cerr << "ERROR (setHitGridCell): Grid index in x-direction was specified as " << ijk.z << ", which is larger than the maximum value based on the grid dimension of " << gridresolution.z-1 << "." << std::endl;
    exit(EXIT_FAILURE);
  }

  hits.at(index).gridcell = ijk;
  
}

void AerialLiDARcloud::coordinateShift( const helios::vec3 shift ){

  if( printmessages ){
    std::cout << "Performing coordinate shift..." << std::flush;
  }

  for( uint r=0; r<hits.size(); r++ ){
    hits.at(r).position = hits.at(r).position + shift;
  }

   if( printmessages ){
     std::cout << "done." << std::endl;
   }
  
}

void AerialLiDARcloud::addHitsToVisualizer( Visualizer* visualizer, const uint pointsize ) const{
  addHitsToVisualizer( visualizer, pointsize, "" );
}

void AerialLiDARcloud::addHitsToVisualizer( Visualizer* visualizer, const uint pointsize, const char* color_value ) const{

  if( printmessages && scans.size()==0 ){
    std::cout << "WARNING (addHitsToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
    return;
  }

  int Ngrid = gridresolution.x*gridresolution.y*gridresolution.z;

  //-- hit points --//
  float minval = 1e9;
  float maxval = -1e9;
  if( strcmp(color_value,"gridcell")==0 ){
    minval = 0;
    maxval = Ngrid-1;
  }else if( strcmp(color_value,"")!=0 ){
    for( uint i=0; i<getHitCount(); i++ ){
      if( doesHitDataExist(i,color_value) ){
	float data = getHitData(i,color_value);
	if( data<minval ){
	  minval = data;
	}
	if( data>maxval ){
	  maxval = data;
	}
      }
    }
  }

  RGBcolor color;
  Colormap cmap = visualizer->getCurrentColormap();
  if( minval!=1e9 && maxval!=-1e9 ){
    cmap.setRange(minval,maxval);
  }

  for( uint i=0; i<getHitCount(); i++ ){
    
    if( strcmp(color_value,"")==0 ){
      color = getHitColor(i);
    }else if( strcmp(color_value,"gridcell")==0 ){
      if( gridijk2index(getHitGridCell(i))<0 ){
	color = RGB::red;
      }else{
	color = cmap.query( gridijk2index(getHitGridCell(i)) );
      }
    }else{
      if( !doesHitDataExist(i,color_value) ){
	color = RGB::red;
      }else{
	float data = getHitData(i,color_value);
	color = cmap.query( data );
      }
    }

    vec3 center = getHitXYZ(i);

    visualizer->addPoint( center, color, pointsize, Visualizer::COORDINATES_CARTESIAN );
    
  }

}

void AerialLiDARcloud::addGridToVisualizer( Visualizer* visualizer ) const{

  if( printmessages && scans.size()==0 ){
    std::cout << "WARNING (addGridToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
    return;
  }

  int Ngrid = gridresolution.x*gridresolution.y*gridresolution.z;

  float minval = 1e9;
  float maxval = -1e9;
  for( uint i=0; i<Ngrid; i++ ){
    float data = getCellLeafAreaDensity(gridindex2ijk(i));
    if( data<minval ){
      minval = data;
    }
    if( data>maxval ){
      maxval = data;
    }
  }

  Colormap cmap = visualizer->getCurrentColormap();
  if( minval!=1e9 && maxval!=-1e9 ){
    cmap.setRange(minval,maxval);
  }

  vec3 gridsize = getCellSize();

  vec3 origin;
  for( uint i=0; i<Ngrid; i++ ){

    //if( getCellLeafAreaDensity(i)==0 ){continue;}

    int3 index = gridindex2ijk(i);
    
    vec3 center = getCellCenter(index);
    
    //RGBAcolor color = make_RGBAcolor(RGB::green,0.5);

    RGBAcolor color = make_RGBAcolor(cmap.query(getCellLeafAreaDensity(index)),0.5);

    visualizer->addVoxelByCenter( center, gridsize, make_SphericalCoord(0,gridrotation), color, Visualizer::COORDINATES_CARTESIAN );

    origin = origin + center/float(Ngrid);
 
  }

  vec3 boxmin, boxmax;
  getHitBoundingBox(boxmin,boxmax);

  float R = 2.f*sqrt( pow(boxmax.x-boxmin.x,2) + pow(boxmax.y-boxmin.y,2) + pow(boxmax.z-boxmin.z,2) );
  
}
  
void AerialLiDARcloud::getHitBoundingBox( helios::vec3& boxmin, helios::vec3& boxmax ) const{

  if( printmessages && hits.size()==0 ){
    std::cout << "WARNING (getHitBoundingBox): There are no hit points in the point cloud, cannot determine bounding box...skipping." << std::endl;
    return;
  }

  boxmin = make_vec3( 1e6, 1e6, 1e6 );
  boxmax = make_vec3( -1e6, -1e6, -1e6 );

  for( std::size_t i=0; i<hits.size(); i++ ){
 
    vec3 xyz = getHitXYZ(i);

    if( xyz.x<boxmin.x ){
      boxmin.x = xyz.x;
    }
    if( xyz.x>boxmax.x ){
      boxmax.x = xyz.x;
    }
    if( xyz.y<boxmin.y ){
      boxmin.y = xyz.y;
    }
    if( xyz.y>boxmax.y ){
      boxmax.y = xyz.y;
    }
    if( xyz.z<boxmin.z ){
      boxmin.z = xyz.z;
    }
    if( xyz.z>boxmax.z ){
      boxmax.z = xyz.z;
    }

  }
    
}

// -------- GRID -------- //

helios::int3 AerialLiDARcloud::gridindex2ijk( const int index ) const{

  if( index<0 || index>=gridresolution.x*gridresolution.y*gridresolution.z ){
    return make_int3(-1,-1,-1);
  }else{

    int k=floor(double(index)/double(gridresolution.x)/double(gridresolution.y));
    int j=floor((double(index)-double(k)*double(gridresolution.x)*double(gridresolution.y))/double(gridresolution.x));
    int i=index-k*gridresolution.x*gridresolution.y-j*gridresolution.x;

    if( i>=0 && i<gridresolution.x && j>=0 && j<gridresolution.y && k>=0 && k<gridresolution.z ){
    }else{
      printf("resolution: %d x %d x %d\n",gridresolution.x,gridresolution.y,gridresolution.z);
      printf("index: %d\n",index);
      printf("ijk: (%d,%d,%d)\n",i,j,k);
    }
    
    assert( i>=0 && i<gridresolution.x && j>=0 && j<gridresolution.y && k>=0 && k<gridresolution.z );
    
    return make_int3(i,j,k);

  }
    
}

int AerialLiDARcloud::gridijk2index( const helios::int3 ijk ) const{

  int index = ijk.z*gridresolution.x*gridresolution.y+ijk.y*gridresolution.y+ijk.x;

  if( index<0 || index>=gridresolution.x*gridresolution.y*gridresolution.z ){
    return -1;
  }else{
    return index;
  }

}

helios::vec3 AerialLiDARcloud::getGridCenter( void ) const{
  
  return gridcenter;
}

helios::vec3 AerialLiDARcloud::getGridExtent( void ) const{
  
  return gridextent;

}

helios::int3 AerialLiDARcloud::getGridResolution( void ) const{
  
  return gridresolution;
}

float AerialLiDARcloud::getGridRotation( void ) const{

  return gridrotation;

}

helios::vec3 AerialLiDARcloud::getCellCenter( const helios::int3 ijk ) const{

  float x = -0.5*gridextent.x+(float(ijk.x)+0.5f)*gridextent.x/float(gridresolution.x);
  float y = -0.5*gridextent.y+(float(ijk.y)+0.5f)*gridextent.y/float(gridresolution.y);

  vec2 xy = rotatePoint2D( make_vec2(x,y), gridrotation ) + make_vec2(gridcenter.x,gridcenter.y);
  
  float z = gridcenter.z-0.5*gridextent.z+(float(ijk.z)+0.5f)*gridextent.z/float(gridresolution.z);

  if( groundheightcomputed ){

    z = z + getCellGroundHeight( make_int2(ijk.x,ijk.y) );
    
  }

  return make_vec3(xy.x,xy.y,z);
  
}

vec3 AerialLiDARcloud::getCellSize( void ) const{

  return make_vec3( gridextent.x/float(gridresolution.x), gridextent.y/float(gridresolution.y), gridextent.z/float(gridresolution.z) );
  
}

// -------- LEAF AREA -------- //

void AerialLiDARcloud::setCellLeafArea( const float area, const int3 ijk ){

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (setCellLeafArea): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (setCellLeafArea): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (setCellLeafArea): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( leaf_area.size()!=gridresolution.z ){
    leaf_area.resize(gridresolution.z);
  }
  if( leaf_area.at(ijk.z).size()!=gridresolution.y ){
    leaf_area.at(ijk.z).resize(gridresolution.y);
  }
  if( leaf_area.at(ijk.z).at(ijk.y).size()!=gridresolution.x ){
    leaf_area.at(ijk.z).at(ijk.y).resize(gridresolution.x);
  }

  leaf_area.at(ijk.z).at(ijk.y).at(ijk.x) = area;

}

float AerialLiDARcloud::getCellLeafArea( const int3 ijk ) const{

  if( leaf_area.size()!=gridresolution.z ){
    cerr << "ERROR (getCellLeafArea): Leaf area calculations have not been run yet. You must first call the function calculateLeafAreaGPU()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (getCellLeafArea): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (getCellLeafArea): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (getCellLeafArea): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  return leaf_area.at(ijk.z).at(ijk.y).at(ijk.x);

}

float AerialLiDARcloud::getCellLeafAreaDensity( const int3 ijk ) const{

  if( leaf_area.size()!=gridresolution.z ){
    cerr << "ERROR (getCellLeafAreaDensity): Leaf area calculations have not been run yet. You must first call the function calculateLeafAreaGPU()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (getCellLeafAreaDensity): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (getCellLeafAreaDensity): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (getCellLeafAreaDensity): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  
  vec3 gridsize( gridextent.x/float(gridresolution.x), gridextent.y/float(gridresolution.y), gridextent.z/float(gridresolution.z) );
  
  return leaf_area.at(ijk.z).at(ijk.y).at(ijk.x)/(gridsize.x*gridsize.y*gridsize.z);

}

void AerialLiDARcloud::setCellTransmissionProbability( const int P_denom, const int P_trans, const int3 ijk ){

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (setCellTransmissionProbability): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (setCellTransmissionProbability): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (setCellTransmissionProbability): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( P_counts.size()!=gridresolution.z ){
    P_counts.resize(gridresolution.z);
  }
  if( P_counts.at(ijk.z).size()!=gridresolution.y ){
    P_counts.at(ijk.z).resize(gridresolution.y);
  }
  if( P_counts.at(ijk.z).at(ijk.y).size()!=gridresolution.x ){
    P_counts.at(ijk.z).at(ijk.y).resize(gridresolution.x);
  }

  P_counts.at(ijk.z).at(ijk.y).at(ijk.x) = make_int2(P_denom,P_trans);

}

void AerialLiDARcloud::getCellTransmissionProbability( const int3 ijk, int& P_denom, int& P_trans ) const{

  if( P_counts.size()!=gridresolution.z ){
    cerr << "ERROR (getCellTransmissionProbability): Leaf area calculations have not been run yet. You must first call the function calculateLeafAreaGPU()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (getCellTransmissionProbabilit): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (getCellTransmissionProbabilit): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (getCellTransmissionProbabilit): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  
  
  P_denom = P_counts.at(ijk.z).at(ijk.y).at(ijk.x).x;
  P_trans = P_counts.at(ijk.z).at(ijk.y).at(ijk.x).y;

}

void AerialLiDARcloud::setCellRbar( const float rbar, const int3 ijk ){

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (setCellRbar): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (setCellRbar): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (setCellRbar): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( r_bar.size()!=gridresolution.z ){
    r_bar.resize(gridresolution.z);
  }
  if( r_bar.at(ijk.z).size()!=gridresolution.y ){
    r_bar.at(ijk.z).resize(gridresolution.y);
  }
  if( r_bar.at(ijk.z).at(ijk.y).size()!=gridresolution.x ){
    r_bar.at(ijk.z).at(ijk.y).resize(gridresolution.x);
  }

  r_bar.at(ijk.z).at(ijk.y).at(ijk.x) = rbar;

}

float AerialLiDARcloud::getCellRbar( const int3 ijk ) const{

  if( r_bar.size()!=gridresolution.z ){
    cerr << "ERROR (getCellRbar): Leaf area calculations have not been run yet. You must first call the function calculateLeafAreaGPU()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ijk.x<0 || ijk.x>=gridresolution.x ){
    cerr << "ERROR (getCellRbar): Cell index in x-direction of " << ijk.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.y<0 || ijk.y>=gridresolution.y ){
    cerr << "ERROR (getCellRbar): Cell index in y-direction of " << ijk.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ijk.z<0 || ijk.z>=gridresolution.z ){
    cerr << "ERROR (getCellRbar): Cell index in z-direction of " << ijk.z << " is outside the allowable range of 0 to " << gridresolution.z-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  
  return r_bar.at(ijk.z).at(ijk.y).at(ijk.x);

}

helios::vec4 AerialLiDARcloud::RANSAC( const int maxIter, const float threshDist, const float inlierRatio, const std::vector<helios::vec3>& hits, std::vector<bool>& inliers ){

  int N = hits.size();

  //trying to find a model in the form of Ax+By+Cz+D=0
  //bestModel.x = A, bestModel.y = B, bestModel.z = C, bestModel.w = D 
  helios::vec4 bestModel;
  float bestError = 1e20;
  
  for( int iter=0; iter<maxIter; iter++ ){

    helios::vec4 maybeModel;
	
    //generate 3 random points
    int i0 = randu(0,N-1);
    int i1 = randu(0,N-1);
    int i2 = randu(0,N-1);

    //get x,y,z coordinates for these points
    helios::vec3 x0 = hits.at(i0);
    helios::vec3 x1 = hits.at(i1);
    helios::vec3 x2 = hits.at(i2);
    
    //determine 'maybeModel' for these three points
    helios::vec3 n = cross( x2-x0, x1-x0 );
    maybeModel.x = n.x;
    maybeModel.y = n.y;
    maybeModel.z = n.z;
    
    maybeModel.w = -(n*x2);

    //loop over all other points
    float errsum = 0.f;
    int Ninlier = 0;
    for( int p=0; p<N; p++ ){
      
      if( p!=i0 && p!=i1 && p!=i2 ){

	helios::vec3 point = hits.at(p);
	    
	float zplane = -(maybeModel.x*point.x+maybeModel.y*point.y+maybeModel.w)/maybeModel.z;
	float zerr = fabs( zplane - point.z ); //NOTE: since we know the plane will be somewhat close to horizontal, we're only using the vertical distance to calculate the error of fit.

	if( zerr<=threshDist ){
	  errsum += zerr;
	  Ninlier += 1;
	}
	
      }

    }

    //check if this model is better than the current 'best' model
    if( errsum/float(Ninlier)<bestError && float(Ninlier)/float(N)>inlierRatio ){

      //update the 'best' model
      bestModel = maybeModel;
      bestError = errsum/float(Ninlier);
      
    }

  }//end iterations loop (iter)

  //separate inliers and outliers based on 'best' model
  inliers.resize(N,0);
  for( int p=0; p<N; p++ ){

    helios::vec3 point = hits.at(p);
	    
    float zplane = -(bestModel.x*point.x+bestModel.y*point.y+bestModel.w)/bestModel.z;
    float zerr = fabs( zplane - point.z );
    
    if( zerr<=threshDist ){
      inliers.at(p) = true;
    }
      
  }
  
  return bestModel;
  
}

void AerialLiDARcloud::generateHeightModel( const int maxIter, const float threshDist_ground, const float inlierRatio_ground, const float threshDist_vegetation, const float inlierRatio_vegetation ){

  if( printmessages ){
    std::cout << "Calculating ground and vegetation heights..." << std::flush;
  }
   
  if( !hitgridcellcomputed ){
    calculateHitGridCell();
  }

  std::srand(time(NULL));

  const uint Nscans = getScanCount();

  helios::int3 gsize3 = getGridResolution();
  const uint Ncells = gsize3.x*gsize3.y*gsize3.z;
  size_t Nhits = getHitCount();

  if( Ncells==0 ){
    std::cerr << "ERROR (generateHeightModel): There are no grid cells in the point cloud." << std::endl;
    exit(EXIT_FAILURE);
  }else if( Nhits==0 ){
    std::cerr << "ERROR (generateHeightModel): There are no hits in the point cloud." << std::endl;
    exit(EXIT_FAILURE);
  }

  //aggregate all points in a given x,y column

  std::vector<std::vector<std::vector<int> > > first_hits_ind; //first index are hit indices, second index is grid cell in x-dir, third index is grid cell in y-dir
  std::vector<std::vector<std::vector<int> > > last_hits_ind; //first index are hit indices, second index is grid cell in x-dir, third index is grid cell in y-dir
  std::vector<std::vector<std::vector<float> > > max_height_xy;
  
  first_hits_ind.resize(gsize3.y);
  last_hits_ind.resize(gsize3.y);
  for( int j=0; j<gsize3.y; j++ ){
    first_hits_ind.at(j).resize(gsize3.x);
    last_hits_ind.at(j).resize(gsize3.x);
  }

  max_height_xy.resize(3);
  for( int k=0; k<max_height_xy.size(); k++ ){
    max_height_xy.at(k).resize(gsize3.y);
    for( int j=0; j<gsize3.y; j++ ){
      max_height_xy.at(k).at(j).resize(gsize3.x);
      for( int i=0; i<gsize3.x; i++ ){
	max_height_xy.at(k).at(j).at(i) = -9999999;
      }
    }
  }

  //do an initial pass to see whether "target_index" starts at 0 or 1
  float target_offset = 0;
  for( size_t r=0; r<Nhits; r++ ){

    if( doesHitDataExist(r,"target_index") ){
      float target_index = getHitData(r,"target_index");
      if( target_index==0 ){
	target_offset=1;
	break;
      }
    }

  }

  for( size_t r=0; r<Nhits; r++ ){

    setHitData( r, "ground_flag", 0 );

    if( !doesHitDataExist(r,"target_index") || !doesHitDataExist(r,"target_count") ){
	continue;
    }

    helios::int3 index3 = getHitGridCell( r );

    if( index3.x>=0 && index3.y>=0 ){

      float target_index = getHitData(r,"target_index")+target_offset;
      float target_count = getHitData(r,"target_count");
      
      float height = getHitXYZ(r).z;

      for( int k=max_height_xy.size()-1; k>=0; k-- ){
	if( height>max_height_xy.at(k).at(index3.y).at(index3.x) ){//&& target_index==1 ){
	  max_height_xy.at(k).at(index3.y).at(index3.x) = height;
	  break;
	}
      }

      if( target_index==target_count ){
	last_hits_ind.at(index3.y).at(index3.x).push_back( r );
      }else if( target_index==1 ){
	first_hits_ind.at(index3.y).at(index3.x).push_back( r );
      }

    }

  }

  //apply RANSAC to first and last hits for each x,y colum to determine ground and vegetation height

  std::vector<std::vector<float> > ground_height_xy, canopy_height_xy;

  ground_height_xy.resize(gsize3.y);
  canopy_height_xy.resize(gsize3.y);

  for( int j=0; j<gsize3.y; j++ ){
    
    ground_height_xy.at(j).resize(gsize3.x);
    canopy_height_xy.at(j).resize(gsize3.x);
    
    for( int i=0; i<gsize3.x; i++ ){

      float canopy_height = nanf("");
      float ground_height = nanf("");

      ground_height_xy.at(j).at(i) = ground_height;
      canopy_height_xy.at(j).at(i) = canopy_height;

      if( last_hits_ind.at(j).at(i).size()==0 ){ //no hits in this column
	continue;
      }

      helios::int3 index3 = getHitGridCell( last_hits_ind.at(j).at(i).front() );

      helios::vec3 center = getCellCenter( index3 ); 

      //compute ground height
      if( last_hits_ind.at(j).at(i).size()>3 ){
	
	std::vector<helios::vec3> last_hits_xyz;
	last_hits_xyz.resize( last_hits_ind.at(j).at(i).size() );
	for( int r=0; r<last_hits_xyz.size(); r++ ){
	  last_hits_xyz.at(r) = getHitXYZ( last_hits_ind.at(j).at(i).at(r) );
	}   
	
	//ground planes
	std::vector<bool> ground_inliers;
	helios::vec4 groundModel = RANSAC( maxIter, threshDist_ground, inlierRatio_ground, last_hits_xyz, ground_inliers );
	
	for( int r=0; r<ground_inliers.size(); r++ ){
	  if( ground_inliers.at(r) ){
	    setHitData( last_hits_ind.at(j).at(i).at(r), "ground_flag", float(1) ); 
	  }
	}

	ground_height = -(groundModel.x*center.x+groundModel.y*center.y+groundModel.w)/groundModel.z;

      }

      //compute canopy height
      if( first_hits_ind.at(j).at(i).size()>=3 ){

	std::vector<helios::vec3> first_hits_xyz;
	first_hits_xyz.resize( first_hits_ind.at(j).at(i).size() );
	for( int r=0; r<first_hits_xyz.size(); r++ ){
	  first_hits_xyz.at(r) = getHitXYZ( first_hits_ind.at(j).at(i).at(r) );
	}

	//canopy top planes
	std::vector<bool> canopy_inliers;
	helios::vec4 canopyModel = RANSAC( maxIter, threshDist_vegetation, inlierRatio_vegetation, first_hits_xyz, canopy_inliers );

	canopy_height = -(canopyModel.x*center.x+canopyModel.y*center.y+canopyModel.w)/canopyModel.z;

	if( canopy_height<ground_height ){
	  canopy_height = ground_height;
	}else if( canopy_height>max_height_xy.front().at(j).at(i) ){
	  canopy_height = max_height_xy.front().at(j).at(i);
	}
	  
      }

      ground_height_xy.at(j).at(i) = ground_height;
      canopy_height_xy.at(j).at(i) = canopy_height;
      
    }//end i (x,y) loop  
  }//end j (x,y) loop

  //fill any holes in the ground height model
  std::vector<std::vector<float> > ground_height_filled;
  ground_height_filled.resize(gsize3.y);
  for( int j=0; j<gsize3.y; j++ ){
    ground_height_filled.at(j).insert(ground_height_filled.at(j).begin(), ground_height_xy.at(j).begin(), ground_height_xy.at(j).end() );
  }
  
  for( int j=1; j<gsize3.y-1; j++ ){
    for( int i=1; i<gsize3.x-1; i++ ){

      if( ground_height_xy.at(j).at(i)!=ground_height_xy.at(j).at(i) ){ //NaN

	int count=0;
	float mean=0;
	for( int jj=-1; jj<2; jj+=2 ){
	  for( int ii=-1; ii<2; ii+=2 ){
	    if( ground_height_xy.at(j+jj).at(i+ii)==ground_height_xy.at(j+jj).at(i+ii) ){
	      mean += ground_height_xy.at(j+jj).at(i+ii);
	      count ++;
	    }
	  }
	}

	if( count>0 ){
	  ground_height_filled.at(j).at(i) = mean/float(count);
	}
	  
      }
      
    }
  }

  //remove outliers in the ground height model
  int stencil_half_size = 4;
  float std_fact = 0.5;

  for( int j=0; j<gsize3.y; j++ ){
    for( int i=0; i<gsize3.x; i++ ){
      ground_height_xy.at(j).at(i) = ground_height_filled.at(j).at(i);
    }
  }
  
  for( int j=stencil_half_size; j<gsize3.y-stencil_half_size; j++ ){
    for( int i=stencil_half_size; i<gsize3.x-stencil_half_size; i++ ){

      if( ground_height_filled.at(j).at(i)!=ground_height_filled.at(j).at(i) ){
	continue;
      }

      int count=0;
      float mean=0;
      float var=0;
      for( int jj=-stencil_half_size; jj<=stencil_half_size; jj++ ){
	for( int ii=-stencil_half_size; ii<=stencil_half_size; ii++ ){
	  if( ii==0 || jj==0 ){
	    continue;
	  }

	  if( ground_height_filled.at(j+jj).at(i+ii)==ground_height_filled.at(j+jj).at(i+ii) ){
	    mean += ground_height_filled.at(j+jj).at(i+ii);
	    var += pow( ground_height_filled.at(j+jj).at(i+ii)-ground_height_filled.at(j).at(i), 2);
	    count ++;
	  }
	}
      }

      float std = sqrt(var/float(count));
      mean = mean/float(count);

      if( fabs(ground_height_filled.at(j).at(i)-mean)>std_fact*std ){
            
	float mean_new=0;
	count = 0;
	for( int jj=-stencil_half_size; jj<=stencil_half_size; jj++ ){
	  for( int ii=-stencil_half_size; ii<=stencil_half_size; ii++ ){
	    if( ii==0 || jj==0 ){
	      continue;
	    }
	    if( fabs(ground_height_filled.at(j+jj).at(i+ii)-mean)<std_fact*std ){
	      mean_new += ground_height_filled.at(j+jj).at(i+ii);
	      count ++;
	    }
	  }
	}
        
	if( count>0 ){
	  ground_height_xy.at(j).at(i) = mean_new/float(count);
	}
      }
    }
  }

  //set the values for all grid cells
  for( int k=0; k<gsize3.z; k++ ){
    for( int j=0; j<gsize3.y; j++ ){
      for( int i=0; i<gsize3.x; i++ ){

	helios::int2 cell(i,j);

	setCellGroundHeight( ground_height_xy.at(j).at(i), cell );
	setCellVegetationHeight( canopy_height_xy.at(j).at(i), cell );
	if( max_height_xy.front().at(j).at(i)==-9999999 ){
	  setCellMaximumHitHeight( nanf(""), cell );
	}else{
	  setCellMaximumHitHeight( max_height_xy.front().at(j).at(i), cell );
	}
	
      }
    }
  }

  groundheightcomputed = true;

  if( printmessages ){
    std::cout << "done." << std::endl;
  }

}


void AerialLiDARcloud::setCellVegetationHeight( const float height, const int2 ij ){

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (setCellVegetationHeight): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (setCellVegetationHeight): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( vegetation_height.size()!=gridresolution.y ){
    vegetation_height.resize(gridresolution.y);
  }
  if( vegetation_height.at(ij.y).size()!=gridresolution.x ){
    vegetation_height.at(ij.y).resize(gridresolution.x);
  }

  vegetation_height.at(ij.y).at(ij.x) = height;
  
}

float AerialLiDARcloud::getCellVegetationHeight( const int2 ij ) const{

  if( vegetation_height.size()!=gridresolution.y ){
    cerr << "ERROR (getCellVegetationHeight): Height calculations have not been run yet. You must first call the function generateHeightModel()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (getCellVegetationHeight): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (getCellVegetationHeight): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  return vegetation_height.at(ij.y).at(ij.x);
  
}

void AerialLiDARcloud::setCellMaximumHitHeight( const float height, const int2 ij ){

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (setCellMaximumHitHeight): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (setCellMaximumHitHeight): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( maxhit_height.size()!=gridresolution.y ){
    maxhit_height.resize(gridresolution.y);
  }
  if( maxhit_height.at(ij.y).size()!=gridresolution.x ){
    maxhit_height.at(ij.y).resize(gridresolution.x);
  }

  maxhit_height.at(ij.y).at(ij.x) = height;
  
}

float AerialLiDARcloud::getCellMaximumHitHeight( const int2 ij ) const{

  if( maxhit_height.size()!=gridresolution.y ){
    cerr << "ERROR (getCellMaximumHitHeight): Height calculations have not been run yet. You must first call the function generateHeightModel()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (getCellMaximumHitHeight): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (getCellMaximumHitHeight): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  return maxhit_height.at(ij.y).at(ij.x);
  
}

void AerialLiDARcloud::setCellGroundHeight( const float height, const int2 ij ){

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (setCellGroundHeight): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (setCellGroundHeight): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ground_height.size()!=gridresolution.y ){
    ground_height.resize(gridresolution.y);
  }
  if( ground_height.at(ij.y).size()!=gridresolution.x ){
    ground_height.at(ij.y).resize(gridresolution.x);
  }

  ground_height.at(ij.y).at(ij.x) = height;
  
}

float AerialLiDARcloud::getCellGroundHeight( const int2 ij ) const{

  if( ground_height.size()!=gridresolution.y ){
    cerr << "ERROR (getCellGroundHeight): Height calculations have not been run yet. You must first call the function generateHeightModel()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (getCellGroundHeight): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (getCellGroundHeight): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  return ground_height.at(ij.y).at(ij.x);
  
}

void AerialLiDARcloud::calculateCoverFraction( void ){

  size_t Nhits = getHitCount();

  std::vector<std::vector<int> > ground_hits, total_hits;

  ground_hits.resize(gridresolution.y);
  total_hits.resize(gridresolution.y);
  for( int j=0; j<gridresolution.y; j++ ){
    ground_hits.at(j).resize(gridresolution.x,0);
    total_hits.at(j).resize(gridresolution.x,0);
  }

  
  for( int r=0; r<Nhits; r++ ){

    int3 ijk = getHitGridCell(r);

    int2 ij(ijk.x,ijk.y);

    if( ij.x>=0 && ij.x<gridresolution.x && ij.y>=0 && ij.y<gridresolution.y ){

      float ground_flag = getHitData(r,"ground_flag");
      if( ground_flag==1 ){
	ground_hits.at(ij.y).at(ij.x)+=1;
      }
      total_hits.at(ij.y).at(ij.x)+=1;
      
    }

  }

  for( int j=0; j<gridresolution.y; j++ ){
    for( int i=0; i<gridresolution.x; i++ ){

      float fcover;
      if( total_hits.at(j).at(i)>0 ){
	fcover = float(ground_hits.at(j).at(i))/float(total_hits.at(j).at(i));
      }else{
	fcover = 0;
      }

      assert( fcover>=0 && fcover<=1.f );

      setCellCoverFraction( fcover, make_int2(i,j) );

    }
  }


}

void AerialLiDARcloud::setCellCoverFraction( const float fcover, const int2 ij ){

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (setCellCoverFraction): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (setCellCoverFraction): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  if( cover_fraction.size()!=gridresolution.y ){
    cover_fraction.resize(gridresolution.y);
  }
  if( cover_fraction.at(ij.y).size()!=gridresolution.x ){
    cover_fraction.at(ij.y).resize(gridresolution.x);
  }

  cover_fraction.at(ij.y).at(ij.x) = fcover;
  
}

float AerialLiDARcloud::getCellCoverFraction( const int2 ij ) const{

  if( cover_fraction.size()!=gridresolution.y ){
    cerr << "ERROR (getCellCoverFraction): Cover fraction calculations have not been run yet. You must first call the function calculateCoverFraction()." << std::endl;
    exit(EXIT_FAILURE);
  }

  if( ij.x<0 || ij.x>=gridresolution.x ){
    cerr << "ERROR (getCellCoverFraction): Cell index in x-direction of " << ij.x << " is outside the allowable range of 0 to " << gridresolution.x-1 << std::endl;
    exit(EXIT_FAILURE);
  }
  if( ij.y<0 || ij.y>=gridresolution.y ){
    cerr << "ERROR (getCellCoverFraction): Cell index in y-direction of " << ij.y << " is outside the allowable range of 0 to " << gridresolution.y-1 << std::endl;
    exit(EXIT_FAILURE);
  }

  return cover_fraction.at(ij.y).at(ij.x);
  
}



vec2 AerialLiDARcloud::rotatePoint2D( const vec2 point, const float theta ) const{

  float st = sinf(theta);
  float ct = cosf(theta);

  // Setup the rotation matrix, this matrix is based off of the rotation matrix 
  float R[2][2];
  R[0][0] = ct;
  R[0][1] = st;
  R[1][0] = -st;
  R[1][1] = ct;

  vec2 tmp;
  tmp.x = R[0][0] * point.x + R[0][1] * point.y;
  tmp.y = R[1][0] * point.x + R[1][1] * point.y;

  return tmp;
  
}

void AerialLiDARcloud::calculateHitGridCell( void ){

  // ---- Hit Points ---- //

  uint Nscans = getScanCount();

  std::size_t total_hits = getHitCount();

  const uint N = total_hits;

  for( std::size_t r=0; r<N; r++ ){
    
    vec3 hit_xyz = getHitXYZ(r);

    //inverse rotate the point to a coordinate system centered at bottom left of grid
    vec2 XY = make_vec2(hit_xyz.x,hit_xyz.y);
    XY = (XY-make_vec2(gridcenter.x,gridcenter.y));
    XY = rotatePoint2D( XY, -gridrotation )+make_vec2(0.5*gridextent.x, 0.5*gridextent.y);
    
    int cx = floor(XY.x/(gridextent.x/float(gridresolution.x)));
    int cy = floor(XY.y/(gridextent.y/float(gridresolution.y)));

    int cz = floor( (hit_xyz.z-gridcenter.z+0.5*gridextent.z)/(gridextent.z/float(gridresolution.z)));

    if( cz<0 || cz>=gridresolution.z ){
      cz = -1;
    }
    
    if( cx<0 || cx>=gridresolution.x ){
      cx = -1;
      cy = -1;
    }else if( cy<0 || cy>=gridresolution.y ){
      cx = -1;
      cy = -1;
    }

    setHitGridCell( r, make_int3(cx,cy,cz) );

  }

  hitgridcellcomputed = true;


}
