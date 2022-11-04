/** \file "LiDAR.cpp" Primary source file for LiDAR plug-in.
    \author Brian Bailey

    Copyright (C) 2016-2022 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "LiDAR.h"

using namespace std;
using namespace helios;

ScanMetadata::ScanMetadata( const helios::vec3 __origin, const uint __Ntheta, const float __thetaMin, const float __thetaMax, const uint __Nphi, const float __phiMin, const float __phiMax, const float __exitDiameter, const float __beamDivergence, const std::vector<std::string> __columnFormat ){

  //Copy arguments into structure variables
  origin = __origin;
  Ntheta = __Ntheta;
  thetaMin = __thetaMin;
  thetaMax = __thetaMax;
  Nphi = __Nphi;
  phiMin = __phiMin;
  phiMax = __phiMax;
  exitDiameter = __exitDiameter;
  beamDivergence = __beamDivergence;
  columnFormat = __columnFormat;

}

helios::SphericalCoord ScanMetadata::rc2direction( const uint row, const uint column ) const{

  float zenith = thetaMin + (thetaMax-thetaMin)/float(Ntheta)*float(row);
  float elevation = 0.5f*M_PI - zenith;
  float phi = phiMin - (phiMax-phiMin)/float(Nphi)*float(column);
  return make_SphericalCoord(1,elevation,phi);

};

helios::int2 ScanMetadata::direction2rc( const helios::SphericalCoord direction ) const{

  float theta = direction.zenith;
  float phi = direction.azimuth;

  int row = round((theta-thetaMin)/(thetaMax-thetaMin)*float(Ntheta));
  int column = round(fabs(phi-phiMin)/(phiMax-phiMin)*float(Nphi));

  if( row==-1 ){
    row = 0;
  }else if( row==Ntheta ){
    row = Ntheta-1;
  }
  if( column==-1 ){
    column = 0;
  }else if( column==Nphi ){
    column = Nphi-1;
  }

  //assert( row>=0 && row<Ntheta );
  //assert( column>=0 && column<Nphi );

  return helios::make_int2(row,column);

};


LiDARcloud::LiDARcloud(){

  Nhits=0;
  hitgridcellcomputed = false;
  triangulationcomputed = false;
  printmessages = true;

}

LiDARcloud::~LiDARcloud( void )= default;

void LiDARcloud::disableMessages(){
  printmessages = false;
}

void LiDARcloud::enableMessages(){
  printmessages = true;
}

void LiDARcloud::validateRayDirections(){

  for( uint s=0; s<getScanCount(); s++ ){

    for( int j=0; j<getScanSizePhi(s); j++ ){
      for( int i=0; i<getScanSizeTheta(s); i++ ){
	if( getHitIndex(s,i,j)>=0 ){
	  SphericalCoord direction1 = scans.at(s).rc2direction(i,j);
	  SphericalCoord direction2 = cart2sphere(getHitXYZ(getHitIndex(s,i,j))-getScanOrigin(s));
	  SphericalCoord direction3 = getHitRaydir(getHitIndex(s,i,j));


	  float err_theta = max( fabs( direction1.zenith-direction2.zenith), fabs( direction1.zenith-direction3.zenith) );

	  float err_phi = max( fabs( direction1.azimuth-direction2.azimuth), fabs( direction1.azimuth-direction3.azimuth) );

	  if( err_theta>1e-6 || err_phi>1e-6 ){
	    cerr << "ERROR: validation of ray directions failed." << endl;
	  }

	}
      }
    }
  }

}

uint LiDARcloud::getScanCount( void ){
  return scans.size();
}

void LiDARcloud::addScan( ScanMetadata newscan ){

    float epsilon = 1e-5;

    if( newscan.thetaMin<0 ){
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan minimum zenith angle of " << newscan.thetaMin << " is less than 0. Truncating to 0." << std::endl;
        newscan.thetaMin = 0;
    }
    if( newscan.phiMin<0 ){
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan minimum azimuth angle of " << newscan.phiMin << " is less than 0. Truncating to 0." << std::endl;
        newscan.phiMin = 0;
    }
    if( newscan.thetaMax>M_PI+epsilon ){
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan maximum zenith angle of " << newscan.thetaMax << " is greater than pi. Truncating to pi. Did you mistakenly use degrees instead of radians?" << std::endl;
        newscan.thetaMax = M_PI;
    }
    if( newscan.phiMax>2.f*M_PI+epsilon ){
        std::cerr << "WARNING (LiDARcloud::addScan): Specified scan maximum azimuth angle of " << newscan.phiMax << " is greater than 2pi. Truncating to 2pi. Did you mistakenly use degrees instead of radians?" << std::endl;
        newscan.phiMax = 2.f*M_PI;
    }

  //initialize the hit table to `-1' (all misses)
  HitTable<int> table;
  table.resize(newscan.Ntheta,newscan.Nphi,-1);
  hit_tables.push_back( table );

  scans.push_back(newscan);
}

void LiDARcloud::addHitPoint( const uint scanID, const helios::vec3 xyz, const helios::SphericalCoord direction ){

  //default color
  RGBcolor color = make_RGBcolor(1,0,0);

  //empty data
  std::map<std::string, double> data;

  addHitPoint( scanID, xyz, direction, color, data );

}

void LiDARcloud::addHitPoint( const uint scanID, const helios::vec3 xyz, const helios::SphericalCoord direction, const std::map<std::string, double> data ){

  //default color
  RGBcolor color = make_RGBcolor(1,0,0);

  addHitPoint( scanID, xyz, direction, color, data );

}

void LiDARcloud::addHitPoint( const uint scanID, const helios::vec3 xyz, const helios::SphericalCoord direction, const helios::RGBcolor color ){

  //empty data
  std::map<std::string, double> data;

  addHitPoint( scanID, xyz, direction, color, data );

}

void LiDARcloud::addHitPoint( const uint scanID, const helios::vec3 xyz, const helios::SphericalCoord direction, const helios::RGBcolor color, const std::map<std::string, double> data ){

  //error checking
  if( scanID>=scans.size() ){
    cerr << "ERROR (addHitPoint): Hit point cannot be added to scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }

  ScanMetadata scan = scans.at(scanID);
  int2 row_column = scan.direction2rc( direction );

  HitPoint hit( scanID, xyz, direction, row_column, color, data );

  hits.push_back(hit);

}

void LiDARcloud::addHitPoint( const uint scanID, const helios::vec3 xyz, const helios::int2 row_column, const helios::RGBcolor color, const std::map<std::string, double> data ){

  ScanMetadata scan = scans.at(scanID);
  SphericalCoord direction = scan.rc2direction( row_column.x, row_column.y );

  HitPoint hit( scanID, xyz, direction, row_column, color, data );

  hits.push_back(hit);

}

void LiDARcloud::deleteHitPoint( const uint index ){

  if( index>=hits.size() ){
    cerr << "WARNING (deleteHitPoint): Hit point #" << index << " cannot be deleted from the scan because there have only been " << hits.size() << " hit points added." << endl;
    return;
  }

  HitPoint hit = hits.at(index);

  int scanID = hit.scanID;

  //erase from vector of hits (use swap-and-pop method)
  std::swap( hits.at(index), hits.back() );
  hits.pop_back();

}

uint LiDARcloud::getHitCount( void ) const{
  return hits.size();
}

helios::vec3 LiDARcloud::getScanOrigin( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanOrigin): Cannot get origin of scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return scans.at(scanID).origin;
}

uint LiDARcloud::getScanSizeTheta( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanSizeTheta): Cannot get theta size for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return scans.at(scanID).Ntheta;
}

uint LiDARcloud::getScanSizePhi( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanSizePhi): Cannot get phi size for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return scans.at(scanID).Nphi;
}

helios::vec2 LiDARcloud::getScanRangeTheta( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanRangeTheta): Cannot get theta range for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return helios::make_vec2(scans.at(scanID).thetaMin,scans.at(scanID).thetaMax);
}

helios::vec2 LiDARcloud::getScanRangePhi( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanRangePhi): Cannot get phi range for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return helios::make_vec2(scans.at(scanID).phiMin,scans.at(scanID).phiMax);
}

float LiDARcloud::getScanBeamExitDiameter( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanBeamExitDiameter): Cannot get exit diameter for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return scans.at(scanID).exitDiameter;
}

float LiDARcloud::getScanBeamDivergence( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanBeamDivergence): Cannot get beam divergence for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return scans.at(scanID).beamDivergence;
}

std::vector<std::string> LiDARcloud::getScanColumnFormat( const uint scanID ) const{
  if( scanID>=scans.size() ){
    cerr << "ERROR (getScanColumnFormat): Cannot get column format for scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  return scans.at(scanID).columnFormat;
}

helios::vec3 LiDARcloud::getHitXYZ( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitXYZ): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  return hits.at(index).position;

}

helios::SphericalCoord LiDARcloud::getHitRaydir( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitRaydir): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  //return hits.at(index).direction;

  vec3 direction_cart = getHitXYZ(index)-getScanOrigin(getHitScanID(index));
  return cart2sphere( direction_cart );

}

void LiDARcloud::setHitData( const uint index, const char* label, const double value ){

  if( index>=hits.size() ){
    cerr << "ERROR (setHitScalarData): Hit point index out of bounds. Tried to set hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  hits.at(index).data[label] = value;

}

double LiDARcloud::getHitData( const uint index, const char* label ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitData): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  std::map<std::string, double> hit_data = hits.at(index).data;
  if( hit_data.find(label) == hit_data.end() ){
    cerr << "ERROR (getHitData): Data value ``" << label << "'' does not exist." << std::endl;
    exit(EXIT_FAILURE);
  }

  return hit_data.at(label);

}

bool LiDARcloud::doesHitDataExist( const uint index, const char* label ) const{

  if( index>=hits.size() ){
    return false;
  }

  std::map<std::string, double> hit_data = hits.at(index).data;
  if( hit_data.find(label) == hit_data.end() ){
    return false;
  }else{
    return true;
  }

}

RGBcolor LiDARcloud::getHitColor( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitColor): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  return hits.at(index).color;

}

int LiDARcloud::getHitScanID( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitColor): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  return hits.at(index).scanID;

}

int LiDARcloud::getHitIndex( const uint scanID, const uint row, const uint column ) const{

  if( scanID>=scans.size() ){
    cerr << "ERROR (deleteHitPoint): Hit point cannot be deleted from scan #" << scanID << " because there have only been " << scans.size() << " scans added." << endl;
    exit(EXIT_FAILURE);
  }
  if( row>=getScanSizeTheta(scanID) ){
    cerr << "ERROR (getHitIndex): Row in scan data table out of range." << endl;
    exit(EXIT_FAILURE);
  }else if( column>=getScanSizePhi(scanID) ){
    cerr << "ERROR (getHitIndex): Column in scan data table out of range." << endl;
    exit(EXIT_FAILURE);
  }

  return hit_tables.at(scanID).get(row,column);
}

int LiDARcloud::getHitGridCell( const uint index ) const{

  if( index>=hits.size() ){
    cerr << "ERROR (getHitGridCell): Hit point index out of bounds. Requesting hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }else if( hits.at(index).gridcell==-2 ){
    cerr << "WARNING (getHitGridCell): hit grid cell for point #" << index << " was never set.  Returning a value of `-1'.  Did you forget to call calculateHitGridCell[*] first?" << endl;
    return -1;
  }

  return hits.at(index).gridcell;

}

void LiDARcloud::setHitGridCell( const uint index, const int cell ){

  if( index>=hits.size() ){
    cerr << "ERROR (setHitGridCell): Hit point index out of bounds. Tried to set hit #" << index << " but scan only has " << hits.size() << " hits." << endl;
    exit(EXIT_FAILURE);
  }

  hits.at(index).gridcell = cell;

}

void LiDARcloud::coordinateShift( const helios::vec3 shift ){

  for( uint s=0; s<scans.size(); s++ ){
    scans.at(s).origin = scans.at(s).origin + shift;
  }

  for( uint r=0; r<hits.size(); r++ ){
    hits.at(r).position = hits.at(r).position + shift;
  }

}

void LiDARcloud::coordinateRotation( const SphericalCoord rotation ){

  for( uint s=0; s<scans.size(); s++ ){
    scans.at(s).origin = rotatePoint(scans.at(s).origin,rotation);
  }

  for( uint r=0; r<hits.size(); r++ ){
    hits.at(r).position = rotatePoint(hits.at(r).position,rotation);
    hits.at(r).direction = cart2sphere(hits.at(r).position - scans.at( hits.at(r).scanID ).origin);
  }

}

void LiDARcloud::coordinateRotation( const float rotation, const helios::vec3 line_base, const helios::vec3 line_direction ){

  for( uint s=0; s<scans.size(); s++ ){
    scans.at(s).origin = rotatePointAboutLine(scans.at(s).origin,line_base,line_direction,rotation);
  }

  for( uint r=0; r<hits.size(); r++ ){
    hits.at(r).position = rotatePointAboutLine(hits.at(r).position,line_base,line_direction,rotation);
    hits.at(r).direction = cart2sphere(hits.at(r).position - scans.at( hits.at(r).scanID ).origin);
  }

}

uint LiDARcloud::getTriangleCount( void ) const{
  return triangles.size();
}

Triangulation LiDARcloud::getTriangle( const uint index ) const{
  if( index>=triangles.size() ){
    cerr << "ERROR (getTriangle): Triangle index out of bounds. Tried to get triangle #" << index << " but point cloud only has " << triangles.size() << " triangles." << endl;
    exit(EXIT_FAILURE);
  }

  return triangles.at(index);

}

void LiDARcloud::addHitsToVisualizer( Visualizer* visualizer, const uint pointsize ) const{
  addHitsToVisualizer( visualizer, pointsize, "" );
}

void LiDARcloud::addHitsToVisualizer( Visualizer* visualizer, const uint pointsize, const char* color_value ) const{

  if( printmessages && scans.size()==0 ){
    std::cout << "WARNING (addHitsToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
    return;
  }

  //-- hit points --//
  float minval = 1e9;
  float maxval = -1e9;
  if( strcmp(color_value,"gridcell")==0 ){
    minval = 0;
    maxval = getGridCellCount()-1;
  }else if( strcmp(color_value,"")!=0 ){
    for( uint i=0; i<getHitCount(); i++ ){
      if( doesHitDataExist(i,color_value) ){
	float data = float(getHitData(i,color_value));
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
      if( getHitGridCell(i)<0 ){
	color = RGB::red;
      }else{
	color = cmap.query( getHitGridCell(i) );
      }
    }else{
      if( !doesHitDataExist(i,color_value) ){
	color = RGB::red;
      }else{
	float data = float(getHitData(i,color_value));
	color = cmap.query( data );
      }
    }

    vec3 center = getHitXYZ(i);

    visualizer->addPoint( center, color, pointsize, Visualizer::COORDINATES_CARTESIAN );

  }

}

void LiDARcloud::addGridToVisualizer( Visualizer* visualizer ) const{

  if( printmessages && scans.size()==0 ){
    std::cout << "WARNING (addGridToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
    return;
  }

  float minval = 1e9;
  float maxval = -1e9;
  for( uint i=0; i<getGridCellCount(); i++ ){
    float data = getCellLeafAreaDensity(i);
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

  vec3 origin;
  for( uint i=0; i<getGridCellCount(); i++ ){

    if( getCellLeafAreaDensity(i)==0 ){continue;}

    vec3 center = getCellCenter(i);

    vec3 anchor = getCellGlobalAnchor(i);

    SphericalCoord rotation = make_SphericalCoord(0,getCellRotation(i));

    center = rotatePointAboutLine( center, anchor, make_vec3(0,0,1), rotation.azimuth );
    vec3 size = getCellSize(i);

    //RGBAcolor color = make_RGBAcolor(RGB::green,0.5);

    RGBAcolor color = make_RGBAcolor(cmap.query(getCellLeafAreaDensity(i)),0.5);

    visualizer->addVoxelByCenter( center, size, rotation, color, Visualizer::COORDINATES_CARTESIAN );

    origin = origin + center/float(getGridCellCount());

  }

  vec3 boxmin, boxmax;
  getHitBoundingBox(boxmin,boxmax);

  float R = 2.f*sqrt( pow(boxmax.x-boxmin.x,2) + pow(boxmax.y-boxmin.y,2) + pow(boxmax.z-boxmin.z,2) );

}

void LiDARcloud::addTrianglesToVisualizer( Visualizer* visualizer ) const{

  if( printmessages && scans.size()==0 ){
    std::cout << "WARNING (addGeometryToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
    return;
  }

  for( uint i=0; i<triangles.size(); i++ ){

    Triangulation tri = triangles.at(i);

    visualizer->addTriangle( tri.vertex0, tri.vertex1, tri.vertex2, tri.color, Visualizer::COORDINATES_CARTESIAN );

  }

}

void LiDARcloud::addTrianglesToVisualizer( Visualizer* visualizer, const uint gridcell ) const{

  if( printmessages && scans.size()==0 ){
    std::cout << "WARNING (addTrianglesToVisualizer): There are no scans in the point cloud, and thus there is no geometry to add...skipping." << std::endl;
    return;
  }

  for( uint i=0; i<triangles.size(); i++ ){

    Triangulation tri = triangles.at(i);

    if( tri.gridcell==gridcell ){
      visualizer->addTriangle( tri.vertex0, tri.vertex1, tri.vertex2, tri.color, Visualizer::COORDINATES_CARTESIAN );
    }

  }

}

void LiDARcloud::addGrid(helios::vec3 gcenter, helios::vec3 gsize, helios::int3 ndiv, float rotation)
{
    if( gsize.x<=0 || gsize.y<=0 || gsize.z<=0 ){
        cerr << "failed.\nERROR (addGrid): The gridcell size must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    if( ndiv.x <=0 || ndiv.y <=0 || ndiv.z <=0 ){
        cerr << "failed.\nERROR (addGrid): The number of grid cells in each direciton must be positive." << endl;
        exit(EXIT_FAILURE);
    }

    //add cells to grid
    vec3 gsubsize = make_vec3(float(gsize.x)/float(ndiv.x),float(gsize.y)/float(ndiv.y),float(gsize.z)/float(ndiv.z));

    float x, y, z;
    uint count = 0;
    for( int k=0; k<ndiv.z; k++ ){
        z = -0.5f*float(gsize.z) + (float(k)+0.5f)*float(gsubsize.z);
        for( int j=0; j<ndiv.y; j++ ){
            y = -0.5f*float(gsize.y) + (float(j)+0.5f)*float(gsubsize.y);
            for( int i=0; i<ndiv.x; i++ ){
                x = -0.5f*float(gsize.x) + (float(i)+0.5f)*float(gsubsize.x);

                vec3 subcenter = make_vec3(x,y,z);

                vec3 subcenter_rot = rotatePoint(subcenter, make_SphericalCoord(0,rotation*M_PI/180.f) );

                if( printmessages ){
                    cout << "Adding grid cell #" << count << " with center " << subcenter_rot.x+gcenter.x << "," << subcenter_rot.y+gcenter.y << "," << subcenter.z+gcenter.z << " and size " << gsubsize.x << " x " << gsubsize.y << " x " << gsubsize.z << endl;
                }

                addGridCell( subcenter+gcenter, gcenter, gsubsize, gsize, rotation*M_PI/180.f, make_int3(i,j,k), ndiv );

                count++;

            }
        }
    }

}

void LiDARcloud::addGridWireFrametoVisualizer(Visualizer* visualizer) const{


    for(int i=0; i< getGridCellCount();i++)
    {
        helios::vec3 center = getCellCenter(i);
        helios::vec3 size = getCellSize(i);

        helios::vec3 boxmin, boxmax;
        boxmin = make_vec3(center.x - 0.5*size.x, center.y - 0.5*size.y, center.z - 0.5*size.z);
        boxmax = make_vec3(center.x + 0.5*size.x, center.y + 0.5*size.y, center.z + 0.5*size.z);

        //vertical edges of the cell
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmin.z), make_vec3(boxmin.x, boxmin.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmax.y, boxmin.z), make_vec3(boxmin.x, boxmax.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmin.y, boxmin.z), make_vec3(boxmax.x, boxmin.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmax.y, boxmin.z), make_vec3(boxmax.x, boxmax.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);

        //horizontal top edges
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmax.z), make_vec3(boxmin.x, boxmax.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmax.z), make_vec3(boxmax.x, boxmin.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmin.y, boxmax.z), make_vec3(boxmax.x, boxmax.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmax.y, boxmax.z), make_vec3(boxmax.x, boxmax.y, boxmax.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);

        //horizontal bottom edges
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmin.z), make_vec3(boxmin.x, boxmax.y, boxmin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmin.y, boxmin.z), make_vec3(boxmax.x, boxmin.y, boxmin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmax.x, boxmin.y, boxmin.z), make_vec3(boxmax.x, boxmax.y, boxmin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);
        visualizer->addLine(make_vec3(boxmin.x, boxmax.y, boxmin.z), make_vec3(boxmax.x, boxmax.y, boxmin.z), RGB::black, 1, Visualizer::COORDINATES_CARTESIAN);

    }


}

void LiDARcloud::addLeafReconstructionToVisualizer( Visualizer* visualizer ) const{

  size_t Ngroups = reconstructed_triangles.size();

  std::vector<helios::RGBcolor> ctable;
  std::vector<float> clocs;

  ctable.push_back( RGB::violet );
  ctable.push_back( RGB::blue );
  ctable.push_back( RGB::green );
  ctable.push_back( RGB::yellow );
  ctable.push_back( RGB::orange );
  ctable.push_back( RGB::red );

  clocs.push_back( 0.f );
  clocs.push_back( 0.2f );
  clocs.push_back( 0.4f );
  clocs.push_back( 0.6f );
  clocs.push_back( 0.8f );
  clocs.push_back( 1.f );

  Colormap colormap( ctable, clocs, 100, 0, Ngroups-1);

  for( size_t g=0; g<Ngroups; g++ ){

    float randi =randu()*(Ngroups-1);
    RGBcolor color = colormap.query(randi);

    for( size_t t=0; t<reconstructed_triangles.at(g).size(); t++ ){

      helios::vec3 v0 = reconstructed_triangles.at(g).at(t).vertex0;
      helios::vec3 v1 = reconstructed_triangles.at(g).at(t).vertex1;
      helios::vec3 v2 = reconstructed_triangles.at(g).at(t).vertex2;

      //RGBcolor color = reconstructed_triangles.at(g).at(t).color;

      visualizer->addTriangle( v0, v1, v2, color, Visualizer::COORDINATES_CARTESIAN );

    }

  }

  Ngroups = reconstructed_alphamasks_center.size();

  for( size_t g=0; g<Ngroups; g++ ){

    visualizer->addRectangleByCenter( reconstructed_alphamasks_center.at(g), reconstructed_alphamasks_size.at(g), reconstructed_alphamasks_rotation.at(g), reconstructed_alphamasks_maskfile.c_str(), Visualizer::COORDINATES_CARTESIAN );

  }

}

void LiDARcloud::addTrunkReconstructionToVisualizer( Visualizer* visualizer ) const{

  size_t Ngroups = reconstructed_trunk_triangles.size();

  for( size_t g=0; g<Ngroups; g++ ){

    for( size_t t=0; t<reconstructed_trunk_triangles.at(g).size(); t++ ){

      helios::vec3 v0 = reconstructed_trunk_triangles.at(g).at(t).vertex0;
      helios::vec3 v1 = reconstructed_trunk_triangles.at(g).at(t).vertex1;
      helios::vec3 v2 = reconstructed_trunk_triangles.at(g).at(t).vertex2;

      RGBcolor color = reconstructed_trunk_triangles.at(g).at(t).color;

      visualizer->addTriangle( v0, v1, v2, color, Visualizer::COORDINATES_CARTESIAN );

    }

  }

}

void LiDARcloud::addTrunkReconstructionToVisualizer( Visualizer* visualizer, const helios::RGBcolor trunk_color ) const{

  size_t Ngroups = reconstructed_trunk_triangles.size();

  for( size_t g=0; g<Ngroups; g++ ){

    for( size_t t=0; t<reconstructed_trunk_triangles.at(g).size(); t++ ){

      helios::vec3 v0 = reconstructed_trunk_triangles.at(g).at(t).vertex0;
      helios::vec3 v1 = reconstructed_trunk_triangles.at(g).at(t).vertex1;
      helios::vec3 v2 = reconstructed_trunk_triangles.at(g).at(t).vertex2;

      visualizer->addTriangle( v0, v1, v2, trunk_color, Visualizer::COORDINATES_CARTESIAN );

    }

  }

}

std::vector<uint> LiDARcloud::addLeafReconstructionToContext( Context* context ) const{
  return addLeafReconstructionToContext( context, helios::make_int2(1,1) );
}

std::vector<uint> LiDARcloud::addLeafReconstructionToContext( Context* context, const helios::int2 subpatches ) const{

  std::vector<uint> UUIDs;

  std::vector<uint> UUID_leaf_template;
  if( subpatches.x>1 || subpatches.y>1 ){
    UUID_leaf_template = context->addTile( make_vec3(0,0,0), make_vec2(1,1), make_SphericalCoord(0,0), subpatches, reconstructed_alphamasks_maskfile.c_str() );
  }

  size_t Ngroups = reconstructed_alphamasks_center.size();

  for( size_t g=0; g<Ngroups; g++ ){

    helios::RGBcolor color = helios::RGB::red;

    uint zone = reconstructed_alphamasks_gridcell.at(g);

    if( reconstructed_alphamasks_size.at(g).x>0 && reconstructed_alphamasks_size.at(g).y>0 ){
      std::vector<uint> UUIDs_leaf;
      if( subpatches.x==1 && subpatches.y==1 ){
	UUIDs_leaf.push_back( context->addPatch( reconstructed_alphamasks_center.at(g), reconstructed_alphamasks_size.at(g), reconstructed_alphamasks_rotation.at(g), reconstructed_alphamasks_maskfile.c_str() ) );
      }else{
	UUIDs_leaf = context->copyPrimitive( UUID_leaf_template );
	context->scalePrimitive( UUIDs_leaf, make_vec3(reconstructed_alphamasks_size.at(g).x,reconstructed_alphamasks_size.at(g).y,1)  );
	context->rotatePrimitive( UUIDs_leaf, -reconstructed_alphamasks_rotation.at(g).elevation, "x" );
	context->rotatePrimitive( UUIDs_leaf, -reconstructed_alphamasks_rotation.at(g).azimuth, "z" );
	context->translatePrimitive( UUIDs_leaf, reconstructed_alphamasks_center.at(g) );
      }
      context->setPrimitiveData( UUIDs_leaf, "gridCell", zone );
      uint flag = reconstructed_alphamasks_direct_flag.at(g);
      context->setPrimitiveData( UUIDs_leaf, "directFlag", flag);
      UUIDs.insert( UUIDs.end(), UUIDs_leaf.begin(), UUIDs_leaf.end() );
    }

  }

  context->deletePrimitive( UUID_leaf_template );

  return UUIDs;

}

std::vector<uint> LiDARcloud::addReconstructedTriangleGroupsToContext( helios::Context* context ) const{

  std::vector<uint> UUIDs;

  size_t Ngroups = reconstructed_triangles.size();

  for( size_t g=0; g<Ngroups; g++ ){

    int leafGroup = round(context->randu()*(Ngroups-1));

    for( size_t t=0; t<reconstructed_triangles.at(g).size(); t++ ){

      helios::vec3 v0 = reconstructed_triangles.at(g).at(t).vertex0;
      helios::vec3 v1 = reconstructed_triangles.at(g).at(t).vertex1;
      helios::vec3 v2 = reconstructed_triangles.at(g).at(t).vertex2;

      RGBcolor color = reconstructed_triangles.at(g).at(t).color;

      UUIDs.push_back( context->addTriangle( v0, v1, v2, color ) );

      uint zone = reconstructed_triangles.at(g).at(t).gridcell;
      context->setPrimitiveData( UUIDs.back(), "gridCell", HELIOS_TYPE_UINT, 1, &zone );

      context->setPrimitiveData( UUIDs.back(), "leafGroup", HELIOS_TYPE_UINT, 1, &leafGroup );

    }

  }

  return UUIDs;

}

std::vector<uint> LiDARcloud::addTrunkReconstructionToContext( Context* context ) const{

  std::vector<uint> UUIDs;

  size_t Ngroups = reconstructed_trunk_triangles.size();

  for( size_t g=0; g<Ngroups; g++ ){

    for( size_t t=0; t<reconstructed_trunk_triangles.at(g).size(); t++ ){

      helios::vec3 v0 = reconstructed_trunk_triangles.at(g).at(t).vertex0;
      helios::vec3 v1 = reconstructed_trunk_triangles.at(g).at(t).vertex1;
      helios::vec3 v2 = reconstructed_trunk_triangles.at(g).at(t).vertex2;

      RGBcolor color = reconstructed_trunk_triangles.at(g).at(t).color;

      UUIDs.push_back( context->addTriangle( v0, v1, v2, color ) );

    }

  }

  return UUIDs;

}

void LiDARcloud::getHitBoundingBox( helios::vec3& boxmin, helios::vec3& boxmax ) const{

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

void LiDARcloud::getGridBoundingBox( helios::vec3& boxmin, helios::vec3& boxmax ) const{

  if( printmessages && getGridCellCount()==0 ){
    std::cout << "WARNING (getGridBoundingBox): There are no grid cells in the point cloud, cannot determine bounding box...skipping." << std::endl;
    return;
  }

  boxmin = make_vec3( 1e6, 1e6, 1e6 );
  boxmax = make_vec3( -1e6, -1e6, -1e6 );

  std::size_t count = 0;
  for( uint c=0; c<getGridCellCount(); c++ ){

    vec3 center = getCellCenter(c);
    vec3 size = getCellSize(c);
    vec3 cellanchor = getCellGlobalAnchor(c);
    float rotation = getCellRotation(c);

    vec3 xyz_min = center - 0.5f*size;
    xyz_min = rotatePointAboutLine(xyz_min,cellanchor,make_vec3(0,0,1),rotation);
    vec3 xyz_max = center + 0.5f*size;
    xyz_max = rotatePointAboutLine(xyz_max,cellanchor,make_vec3(0,0,1),rotation);

    if( xyz_min.x<boxmin.x ){
      boxmin.x = xyz_min.x;
    }
    if( xyz_max.x>boxmax.x ){
      boxmax.x = xyz_max.x;
    }
    if( xyz_min.y<boxmin.y ){
      boxmin.y = xyz_min.y;
    }
    if( xyz_max.y>boxmax.y ){
      boxmax.y = xyz_max.y;
    }
    if( xyz_min.z<boxmin.z ){
      boxmin.z = xyz_min.z;
    }
    if( xyz_max.z>boxmax.z ){
      boxmax.z = xyz_max.z;
    }

  }


}

void LiDARcloud::distanceFilter( const float maxdistance ){
    
    std::size_t delete_count = 0;
    for(int i = (getHitCount()-1); i>=0; i-- ){
        
        vec3 xyz = getHitXYZ(i);
        uint scanID = getHitScanID(i);
        vec3 r = xyz-getScanOrigin(scanID);
        
        if( r.magnitude()>maxdistance ){
            deleteHitPoint(i);
            delete_count++;
        }
    }
    
    if( printmessages ){
        std::cout << "Removed " << delete_count << " hit points based on distance filter." << std::endl;
    }
}

void LiDARcloud::reflectanceFilter( const float minreflectance ){
    
    std::size_t delete_count = 0;
    for(int r = (getHitCount()-1); r>=0; r-- ){  
        if( hits.at(r).data.find("reflectance") != hits.at(r).data.end() ){
            double R = getHitData(r,"reflectance");
            if( R<minreflectance ){
                deleteHitPoint(r);
                delete_count++;
            }
        }
    }
    
    if( printmessages ){
        std::cout << "Removed " << delete_count << " hit points based on reflectance filter." << std::endl;
    }
    
}

void LiDARcloud::scalarFilter( const char* scalar_field, const float threshold, const char* comparator ){
    
    std::size_t delete_count = 0;
    for(int r = (getHitCount()-1); r>=0; r-- ){ 
        if( hits.at(r).data.find(scalar_field) != hits.at(r).data.end() ){
            double R = getHitData(r,scalar_field);
            if( strcmp(comparator,"<")==0 ){
                if( R<threshold ){
                    deleteHitPoint(r);
                    delete_count++;
                }
            }else if( strcmp(comparator,">")==0 ){
                if( R>threshold ){
                    deleteHitPoint(r);
                    delete_count++;
                }
            }else if( strcmp(comparator,"=")==0 ){
                if( R==threshold ){
                    deleteHitPoint(r);
                    delete_count++;
                }
            }
        }
    }
    
    if( printmessages ){
        std::cout << "Removed " << delete_count << " hit points based on scalar filter." << std::endl;
    }
    
}

void LiDARcloud::xyzFilter( const float xmin, const float xmax, const float ymin, const float ymax, const float zmin, const float zmax ){
    
    if(xmin > xmax || ymin > ymax || zmin > zmax)
    {
        std::cout << "WARNING: at least one minimum value provided is greater than one maximum value. " << std::endl; 
    }
    
    std::size_t delete_count = 0;
    
    for(int i = (getHitCount()-1); i>=0; i-- ){  
        std::cout << i << std::endl;
        vec3 xyz = getHitXYZ(i);
        if( xyz.x < xmin || xyz.x > xmax || xyz.y < ymin || xyz.y > ymax || xyz.z < zmin || xyz.z > zmax ){
            deleteHitPoint(i);
            delete_count++;
        }
    }
    
    if( printmessages ){
        std::cout << "Removed " << delete_count << " hit points based on provided bounding box." << std::endl;
    }
    
}

// bool sortcol0( const std::vector<float>& v0, const std::vector<float>& v1 ){
//   return v0.at(0)<v1.at(0);
// }

// bool sortcol1( const std::vector<float>& v0, const std::vector<float>& v1 ){
//   return v0.at(1)<v1.at(1);
// }

bool sortcol0( const std::vector<double>& v0, const std::vector<double>& v1 ){
  return v0.at(0)<v1.at(0);
}

bool sortcol1( const std::vector<double>& v0, const std::vector<double>& v1 ){
  return v0.at(1)<v1.at(1);
}

void LiDARcloud::maxPulseFilter( const char* scalar ){

  if( printmessages ){
    std::cout << "Filtering point cloud by maximum " << scalar << " per pulse..." << std::flush;
  }

  std::vector<std::vector<double> > timestamps;
  timestamps.resize(getHitCount());

  std::size_t delete_count = 0;
  for( std::size_t r=0; r<getHitCount(); r++ ){

    if( !doesHitDataExist(r,"timestamp") ){
      std::cout << "ERROR (maxPulseFilter): Hit point " << r << " does not have scalar data ""timestamp""." << std::endl;
      return;
    }else if( !doesHitDataExist(r,scalar) ){
      std::cout << "ERROR (maxPulseFilter): Hit point " << r << " does not have scalar data """ << scalar << """." << std::endl;
      return;
    }

    std::vector<double> v{getHitData(r,"timestamp"),getHitData(r,scalar),double(r)};

    timestamps.at(r) = v;

  }

  std::sort( timestamps.begin(), timestamps.end(), sortcol0 );

  std::vector<std::vector<double> > isort;
  std::vector<int> to_delete;
  double time_old = timestamps.at(0).at(0);
  for( std::size_t r=0; r<timestamps.size(); r++ ){

    if( timestamps.at(r).at(0)!=time_old ){

      if( isort.size()>1 ){

	std::sort( isort.begin(), isort.end(), sortcol1 );

	for( int i=0; i<isort.size()-1; i++ ){
	  to_delete.push_back( int(isort.at(i).at(2)) );
	}

      }

      isort.resize(0);
      time_old = timestamps.at(r).at(0);
    }

    isort.push_back(timestamps.at(r));

  }

  std::sort( to_delete.begin(), to_delete.end() );

  for( int i=to_delete.size()-1; i>=0; i-- ){
    deleteHitPoint( to_delete.at(i) );
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }


}

void LiDARcloud::minPulseFilter( const char* scalar ){

  if( printmessages ){
    std::cout << "Filtering point cloud by minimum " << scalar << " per pulse..." << std::flush;
  }

  std::vector<std::vector<double> > timestamps;
  timestamps.resize(getHitCount());

  std::size_t delete_count = 0;
  for( std::size_t r=0; r<getHitCount(); r++ ){

    if( !doesHitDataExist(r,"timestamp") ){
      std::cout << "ERROR (minPulseFilter): Hit point " << r << " does not have scalar data ""timestamp""." << std::endl;
      return;
    }else if( !doesHitDataExist(r,scalar) ){
      std::cout << "ERROR (minPulseFilter): Hit point " << r << " does not have scalar data """ << scalar << """." << std::endl;
      return;
    }

    std::vector<double> v{getHitData(r,"timestamp"),getHitData(r,scalar),float(r)};

    timestamps.at(r) = v;

  }

  std::sort( timestamps.begin(), timestamps.end(), sortcol0 );

  std::vector<std::vector<double> > isort;
  std::vector<int> to_delete;
  double time_old = timestamps.at(0).at(0);
  for( std::size_t r=0; r<timestamps.size(); r++ ){

    if( timestamps.at(r).at(0)!=time_old ){

      if( isort.size()>1 ){

	std::sort( isort.begin(), isort.end(), sortcol1 );

	for( int i=1; i<isort.size(); i++ ){
	  to_delete.push_back( int(isort.at(i).at(2)) );
	}

      }

      isort.resize(0);
      time_old = timestamps.at(r).at(0);
    }

    isort.push_back(timestamps.at(r));

  }

  std::sort( to_delete.begin(), to_delete.end() );

  for( int i=to_delete.size()-1; i>=0; i-- ){
    deleteHitPoint( to_delete.at(i) );
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }


}

void LiDARcloud::firstHitFilter( void ){

  if( printmessages ){
    std::cout << "Filtering point cloud to only first hits per pulse..." << std::flush;
  }

  std::vector<float> target_index;
  target_index.resize(getHitCount());
  int min_tindex = 1;

  for( std::size_t r=0; r<target_index.size(); r++ ){

    if( !doesHitDataExist(r,"target_index") ){
      std::cout << "ERROR (firstHitFilter): Hit point " << r << " does not have scalar data ""target_index""." << std::endl;
      return;
    }

    target_index.at(r) = getHitData(r,"target_index");

    if( target_index.at(r) == 0 ){
      min_tindex = 0;
    }

  }

  for( int r=target_index.size()-1; r>=0; r-- ){

    if( target_index.at(r)!=min_tindex ){
      deleteHitPoint(r);
    }

  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }


}

void LiDARcloud::lastHitFilter( void ){

  if( printmessages ){
    std::cout << "Filtering point cloud to only last hits per pulse..." << std::flush;
  }

  std::vector<float> target_index;
  target_index.resize(getHitCount());
  int min_tindex = 1;

  for( std::size_t r=0; r<target_index.size(); r++ ){

    if( !doesHitDataExist(r,"target_index") ){
      std::cout << "ERROR (lastHitFilter): Hit point " << r << " does not have scalar data ""target_index""." << std::endl;
      return;
    }else if( !doesHitDataExist(r,"target_count") ){
      std::cout << "ERROR (lastHitFilter): Hit point " << r << " does not have scalar data ""target_count""." << std::endl;
      return;
    }

    target_index.at(r) = getHitData(r,"target_index");

    if( target_index.at(r) == 0 ){
      min_tindex = 0;
    }

  }

  for( int r=target_index.size()-1; r>=0; r-- ){

    float target_count = getHitData(r,"target_count");

    if( target_index.at(r)==target_count-1+min_tindex ){
      deleteHitPoint(r);
    }

  }

  if( printmessages ){
    std::cout << "done." << std::endl;
  }

}

void LiDARcloud::triangulateHitPoints( const float Lmax, const float max_aspect_ratio ){

  if( printmessages && getScanCount()==0 ){
    cout << "WARNING (triangulateHitPoints): No scans have been added to the point cloud.  Skipping triangulation..." << endl;
    return;
  }else if( printmessages && getHitCount()==0 ){
    cout << "WARNING (triangulateHitPoints): No hit points have been added to the point cloud.  Skipping triangulation..." << endl;
    return;
  }

  if( !hitgridcellcomputed ){
    calculateHitGridCellGPU();
  }

  int Ntriangles=0;

  for( uint s=0; s<getScanCount(); s++ ){

    std::vector<int> Delaunay_inds;

    std::vector<Shx> pts, pts_copy;

    int count = 0;
    for( int r=0; r<getHitCount(); r++ ){

      if( getHitScanID(r)==s && getHitGridCell(r)>=0 ){

	helios::SphericalCoord direction = getHitRaydir(r);

	helios::vec3 direction_cart = getHitXYZ(r)-getScanOrigin(s);
	direction = cart2sphere( direction_cart );

	Shx pt;
	pt.id = count;
	pt.r = direction.zenith;
	pt.c = direction.azimuth;

	pts.push_back(pt);

	Delaunay_inds.push_back(r);

	count++;

      }

    }

    if( pts.size()==0 ){
      if( printmessages ){
	std::cout << "Scan " << s << " contains no triangles. Skipping this scan..." << std::endl;
      }
      continue;
    }

    float h[2] = {0,0};
    for( int r=0; r<pts.size(); r++ ){
      if( pts.at(r).c<0.5*M_PI ){
	h[0] += 1.f;
      }else if( pts.at(r).c>1.5*M_PI ){
	h[1] += 1.f;
      }
    }
    h[0] /= float(pts.size());
    h[1] /= float(pts.size());
    if( h[0]+h[1]>0.4 ){
      if( printmessages ){
	std::cout << "Shifting scan " << s << std::endl;
      }
      for( int r=0; r<pts.size(); r++ ){
	pts.at(r).c += M_PI;
	if( pts.at(r).c > 2.f*M_PI ){
	  pts.at(r).c -= 2.f*M_PI;
	}
      }
    }

    std::vector<int> dupes;
    int nx = de_duplicate( pts, dupes);
    pts_copy = pts;

    std::vector<Triad> triads;

    if( printmessages ){
      std::cout << "starting triangulation for scan " << s << "..." << std::endl;
    }

    int success = 0;
    int Ntries = 0;
    while( success!=1 && Ntries<3 ){
      Ntries++;

      success = s_hull_pro( pts, triads );

      if( success!=1 ){

	//try a 90 degree coordinate shift
	if( printmessages ){
	  std::cout << "Shifting scan " << s << " (try " << Ntries << " of 3)" << std::endl;
	}
	for( int r=0; r<pts.size(); r++ ){
	  pts.at(r).c += 0.25*M_PI;
	  if( pts.at(r).c > 2.f*M_PI ){
	    pts.at(r).c -= 2.f*M_PI;
	  }
	}
      }

    }

    if( success!=1 ){
      if( printmessages ){
	std::cout << "FAILED: could not triangulate scan " << s << ". Skipping this scan." << std::endl;
      }
      continue;
    }else if( printmessages ){
      std::cout << "finished triangulation" << std::endl;
    }

    for( int t=0; t<triads.size(); t++ ){

      int ID0 = Delaunay_inds.at(triads.at(t).a);
      int ID1 = Delaunay_inds.at(triads.at(t).b);
      int ID2 = Delaunay_inds.at(triads.at(t).c);

      helios::vec3 vertex0 = getHitXYZ( ID0 );
      helios::SphericalCoord raydir0 = getHitRaydir( ID0 );

      helios::vec3 vertex1 = getHitXYZ( ID1 );
      helios::SphericalCoord raydir1 = getHitRaydir( ID1 );

      helios::vec3 vertex2 = getHitXYZ( ID2 );
      helios::SphericalCoord raydir2 = getHitRaydir( ID2 );

      helios::vec3 v;
      v=vertex0-vertex1;
      float L0 = v.magnitude();
      v=vertex0-vertex2;
      float L1 = v.magnitude();
      v=vertex1-vertex2;
      float L2 = v.magnitude();

      float aspect_ratio = max(max(L0,L1),L2)/min(min(L0,L1),L2);

      if( L0>Lmax || L1>Lmax || L2>Lmax || aspect_ratio>max_aspect_ratio ){
	continue;
      }

      int gridcell = getHitGridCell( ID0 );

      if( printmessages && gridcell==-2 ){
	cout << "WARNING (triangulateHitPoints): You typically want to define the hit grid cell for all hit points before performing triangulation." << endl;
      }

      RGBcolor color = make_RGBcolor(0,0,0);
      color.r = (hits.at(ID0).color.r + hits.at(ID1).color.r + hits.at(ID2).color.r )/3.f;
      color.g = (hits.at(ID0).color.g + hits.at(ID1).color.g + hits.at(ID2).color.g )/3.f;
      color.b = (hits.at(ID0).color.b + hits.at(ID1).color.b + hits.at(ID2).color.b )/3.f;

      Triangulation tri( s, vertex0, vertex1, vertex2, ID0, ID1, ID2, color, gridcell );

      triangles.push_back(tri);

      Ntriangles++;

    }

  }

  triangulationcomputed = true;

  if( printmessages ){
    cout << "\r                                           " ;
    cout << "\rTriangulating...formed " << Ntriangles << " total triangles." << endl;
  }

}

void LiDARcloud::addTrianglesToContext( Context* context ) const{

  if( scans.size()==0 ){
    if( printmessages ){
      std::cout << "WARNING (addTrianglesToContext): There are no scans in the point cloud, and thus there are no triangles to add...skipping." << std::endl;
    }
    return;
  }

  for( std::size_t i=0; i<getTriangleCount(); i++ ){

    Triangulation tri = getTriangle(i);

    context->addTriangle( tri.vertex0, tri.vertex1, tri.vertex2, tri.color );

  }

}

uint LiDARcloud::getGridCellCount( void ) const{
  return grid_cells.size();
}

void LiDARcloud::addGridCell( const helios::vec3 center, const helios::vec3 size, const float rotation ){
  addGridCell(center,center,size,size,rotation,make_int3(1,1,1), make_int3(1,1,1));
}

void LiDARcloud::addGridCell( const helios::vec3 center, const helios::vec3 global_anchor, const helios::vec3 size, const helios::vec3 global_size, const float rotation, const helios::int3 global_ijk, const helios::int3 global_count ){

  GridCell newcell( center, global_anchor, size, global_size, rotation, global_ijk, global_count );

  grid_cells.push_back(newcell);

}

helios::vec3 LiDARcloud::getCellCenter( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellCenter): grid cell index out of range.  Requested center of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  return grid_cells.at(index).center;

}

helios::vec3 LiDARcloud::getCellGlobalAnchor( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellGlobalAnchor): grid cell index out of range.  Requested anchor of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  return grid_cells.at(index).global_anchor;

}

helios::vec3 LiDARcloud::getCellSize( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellCenter): grid cell index out of range.  Requested size of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  return grid_cells.at(index).size;

}

float LiDARcloud::getCellRotation( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellRotation): grid cell index out of range.  Requested rotation of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  return grid_cells.at(index).azimuthal_rotation;

}

std::vector<float> LiDARcloud::calculateSyntheticGtheta( helios::Context* context ){

  size_t Nprims = context->getPrimitiveCount();

  uint Nscans = getScanCount();

  uint Ncells = getGridCellCount();

  std::vector<float> Gtheta;
  Gtheta.resize(Ncells);

  std::vector<float> area_sum;
  area_sum.resize(Ncells,0.f);
  std::vector<float> sin_sum;
  sin_sum.resize(Ncells,0.f);
  std::vector<uint> cell_tri_count;
  cell_tri_count.resize(Ncells,0);

  std::vector<uint> UUIDs = context->getAllUUIDs();
  for( int p=0; p<UUIDs.size(); p++ ){

    uint UUID = UUIDs.at(p);

    if( context->doesPrimitiveDataExist(UUID,"gridCell") ){

      uint gridCell;
      context->getPrimitiveData(UUID,"gridCell",gridCell);

      std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);
      float area = context->getPrimitiveArea(UUID);
      vec3 normal = context->getPrimitiveNormal(UUID);

      for( int s=0; s<Nscans; s++ ){
	vec3 origin = getScanOrigin(s);
	vec3 raydir = vertices.front()-origin;
	raydir.normalize();
	float theta = fabs(acos_safe(raydir.z));

	if( area==area ){ //in rare cases you can get area=NaN

	  Gtheta.at(gridCell) += fabs(normal*raydir)*area*fabs(sin(theta));

	  area_sum.at(gridCell) += area;
	  sin_sum.at(gridCell) += fabs(sin(theta));
	  cell_tri_count.at(gridCell) += 1;

	}
      }

    }
  }

  for( uint v=0; v<Ncells; v++ ){
    if( cell_tri_count[v]>0 ){
      Gtheta[v] *= float(cell_tri_count[v])/(area_sum[v]*sin_sum[v]);
    }
  }


  std::vector<float> output_Gtheta;
  output_Gtheta.resize(Ncells,0.f);

  for( int v=0; v<Ncells; v++ ){
    output_Gtheta.at(v) = Gtheta.at(v);
    if( context->doesPrimitiveDataExist(UUIDs.at(v),"gridCell") ){
      context->setPrimitiveData(UUIDs.at(v),"synthetic_Gtheta",Gtheta.at(v));
    }
  }

  return output_Gtheta;

}

void LiDARcloud::setCellLeafArea( const float area, const uint index ){

  if( index>getGridCellCount() ){
    cout << "ERROR (setCellLeafArea): grid cell index out of range." << endl;
  }

  grid_cells.at(index).leaf_area = area;

}

float LiDARcloud::getCellLeafArea( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellLeafArea): grid cell index out of range. Requested leaf area of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  return grid_cells.at(index).leaf_area;

}

float LiDARcloud::getCellLeafAreaDensity( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellLeafAreaDensity): grid cell index out of range. Requested leaf area density of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  helios::vec3 gridsize = grid_cells.at(index).size;
  return grid_cells.at(index).leaf_area/(gridsize.x*gridsize.y*gridsize.z);

}

void LiDARcloud::setCellGtheta( const float Gtheta, const uint index ){

  if( index>getGridCellCount() ){
    cout << "ERROR (setCellGtheta): grid cell index out of range." << endl;
  }

  grid_cells.at(index).Gtheta = Gtheta;

}

float LiDARcloud::getCellGtheta( const uint index ) const{

  if( index>=getGridCellCount() ){
    cout << "ERROR (getCellGtheta): grid cell index out of range. Requested leaf area of cell #" << index << " but there are only " << getGridCellCount() << " cells in the grid." << endl;
    exit(EXIT_FAILURE);
  }

  return grid_cells.at(index).Gtheta;

}

void LiDARcloud::leafReconstructionFloodfill( void ){

  size_t group_count = 0;
  int current_group = 0;

  vector<vector<int> > nodes;
  nodes.resize(getHitCount());

  size_t Ntri = 0;
  for( size_t t=0; t<getTriangleCount(); t++ ){

    Triangulation tri = getTriangle(t);

    if( tri.gridcell >=0 ){

      nodes.at( tri.ID0 ).push_back(t);
      nodes.at( tri.ID1 ).push_back(t);
      nodes.at( tri.ID2 ).push_back(t);

      Ntri++;

    }

  }

  std::vector<int> fill_flag;
  fill_flag.resize(Ntri);
  for( size_t t=0; t<Ntri; t++ ){
    fill_flag.at(t)=-1;
  }

  for( size_t t=0; t<Ntri; t++ ){//looping through all triangles

    if( fill_flag.at(t)<0 ){

      floodfill( t, triangles, fill_flag, nodes, current_group, 0, 1e3 );

      current_group ++;

    }

  }

  for( size_t t=0; t<Ntri; t++ ){//looping through all triangles

    if( fill_flag.at(t)>=0 ){
      int fill_group = fill_flag.at(t);

      if( fill_group>=reconstructed_triangles.size() ){
	reconstructed_triangles.resize( fill_group+1 );
      }

      reconstructed_triangles.at(fill_group).push_back(triangles.at(t));

    }

  }

}

void LiDARcloud::floodfill( size_t t, std::vector<Triangulation> &t_triangles, std::vector<int> &fill_flag, std::vector<std::vector<int> > &nodes, const int tag, const int depth, const int maxdepth ){

  Triangulation tri = t_triangles.at(t);

  int verts[3] = {tri.ID0, tri.ID1, tri.ID2};

  std::vector<int> connection_list;

  for( int i=0; i<3; i++ ){
    std::vector<int> connected_tris = nodes.at(verts[i]);
    connection_list.insert( connection_list.begin(), connected_tris.begin(),connected_tris.end());
  }

  std::sort( connection_list.begin(), connection_list.end() );

  int count = 0;
  for( int tt=1; tt<connection_list.size(); tt++ ){
    if( connection_list.at(tt-1)!=connection_list.at(tt) ){

      if( count>=2 ){

  	int index = connection_list.at(tt-1);

  	if( fill_flag.at(index)==-1 && index!=t ){

  	  fill_flag.at(index) = tag;

  	  if( depth<maxdepth ){
  	    floodfill( index, t_triangles, fill_flag, nodes, tag, depth+1, maxdepth );
  	  }

  	}

      }

      count = 1;
    }else{
      count++;
    }

  }


}

void LiDARcloud::leafReconstructionAlphaMask( const float minimum_leaf_group_area, const float maximum_leaf_group_area, const float leaf_aspect_ratio, const char* mask_file ){
  leafReconstructionAlphaMask( minimum_leaf_group_area, maximum_leaf_group_area, leaf_aspect_ratio, -1.f, mask_file );
}

void LiDARcloud::leafReconstructionAlphaMask( const float minimum_leaf_group_area, const float maximum_leaf_group_area, const float leaf_aspect_ratio, const float leaf_length_constant, const char* mask_file ){

  if( printmessages ){
    cout << "Performing alphamask leaf reconstruction..." << flush;
  }

  if( triangles.size()==0 ){
    std::cout << "Performing alphamask leaf reconstruction...failed." << std::endl;
    std::cerr << "There are no triangulated points.  Either the triangulation failed or 'triangulateHitPoints()' was not called." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string file = mask_file;
  if( file.substr(file.find_last_of(".") + 1) != "png") {
    std::cerr << "ERROR (leafReconstructionAlphaMask): Mask data file " << mask_file << " must be PNG image format." << std::endl;
    exit(EXIT_FAILURE);
  }
  std::vector<std::vector<bool> > maskdata = readPNGAlpha(mask_file);
  if( maskdata.size()==0 ){
    std::cerr << "ERROR (leafReconstructionAlphaMask): Could not load mask file " << mask_file << ". It contains no data." << std::endl;
    exit(EXIT_FAILURE);
  }
  int ix = maskdata.front().size();
  int jy = maskdata.size();
  int2 masksize = make_int2(ix,jy);
  uint Atotal=0;
  uint Asolid=0;
  for( uint j=0; j<masksize.y; j++ ){
    for( uint i=0; i<masksize.x; i++ ){
      Atotal++;
      if( maskdata.at(j).at(i) ){
	Asolid++;
      }
    }
  }

  float solidfraction = float(Asolid)/float(Atotal);

  float total_area = 0.f;

  std::vector<std::vector<float> > group_areas;
  group_areas.resize(getGridCellCount());

  reconstructed_alphamasks_maskfile = mask_file;

  leafReconstructionFloodfill();

  //Filter out small groups by an area threshold

  uint group_count = reconstructed_triangles.size();

  float group_area_max = 0;

  std::vector<bool> group_filter_flag;
  group_filter_flag.resize(reconstructed_triangles.size());

  for( int group=group_count-1; group>=0; group-- ){

    float garea = 0.f;

    for( size_t t=0; t<reconstructed_triangles.at(group).size(); t++ ){

      float triangle_area = reconstructed_triangles.at(group).at(t).area;

      garea += triangle_area;

    }

    if( garea<minimum_leaf_group_area || garea>maximum_leaf_group_area ){
      group_filter_flag.at(group) = false;
      //reconstructed_triangles.erase( reconstructed_triangles.begin()+group );
    }else{
      group_filter_flag.at(group) = true;
      int cell = reconstructed_triangles.at(group).front().gridcell;
      group_areas.at(cell).push_back(garea);
    }

  }

  vector<float> Lavg;
  Lavg.resize( getGridCellCount(), 0.f );

  int Navg = 20;

  for( int v=0; v<getGridCellCount(); v++ ){

    std::sort( group_areas.at(v).begin(), group_areas.at(v).end() );
    //std::partial_sort( group_areas.at(v).begin(), group_areas.at(v).begin()+Navg,group_areas.at(v).end(), std::greater<float>() );

    if( group_areas.at(v).size()>Navg ){
      for( int i=group_areas.at(v).size()-1; i>=group_areas.at(v).size()-Navg; i-- ){
	Lavg.at(v) += sqrtf(group_areas.at(v).at(i))/float(Navg);
      }
    }else if( group_areas.at(v).size()==0 ){
      Lavg.at(v) = 0.05; //NOTE: hard-coded
    }else{
      for( int i=0; i<group_areas.at(v).size(); i++ ){
	Lavg.at(v) += sqrtf(group_areas.at(v).at(i))/float(group_areas.at(v).size());
      }
    }

    if( printmessages ){
      std::cout << "Average leaf length for volume #" << v << " : " << Lavg.at(v) << endl;
    }

  }

  //Form alphamasks

  for( int group=0; group<reconstructed_triangles.size(); group++ ){

    if( !group_filter_flag.at(group) ){
      continue;
    }

    int cell = reconstructed_triangles.at(group).front().gridcell;

    helios::vec3 position = make_vec3(0,0,0);
    for( int t=0; t<reconstructed_triangles.at(group).size(); t++ ){
      position = position + reconstructed_triangles.at(group).at(t).vertex0/float(reconstructed_triangles.at(group).size());
    }

    int gind = round( randu()*(reconstructed_triangles.at(group).size()-1) );

    reconstructed_alphamasks_center.push_back( position );
    float l = Lavg.at(reconstructed_triangles.at(group).front().gridcell)*sqrt(leaf_aspect_ratio/solidfraction);
    float w = l/leaf_aspect_ratio;
    reconstructed_alphamasks_size.push_back( helios::make_vec2(w,l) );
    helios::vec3 normal = cross( reconstructed_triangles.at(group).at(gind).vertex1-reconstructed_triangles.at(group).at(gind).vertex0, reconstructed_triangles.at(group).at(gind).vertex2-reconstructed_triangles.at(group).at(gind).vertex0 );
    reconstructed_alphamasks_rotation.push_back( make_SphericalCoord(cart2sphere(normal).zenith,cart2sphere(normal).azimuth)  );
    reconstructed_alphamasks_gridcell.push_back( reconstructed_triangles.at(group).front().gridcell );
    reconstructed_alphamasks_direct_flag.push_back( 1 );
  }

  if( printmessages ){
    cout << "done." << endl;
    cout << "Directly reconstructed " << reconstructed_alphamasks_center.size() << " leaf groups." << endl;
  }

  backfillLeavesAlphaMask( Lavg, leaf_aspect_ratio, solidfraction, group_filter_flag );

  for( int group=0; group<reconstructed_triangles.size(); group++ ){

    if( !group_filter_flag.at(group) ){
      std::swap( reconstructed_triangles.at(group), reconstructed_triangles.back() );
      reconstructed_triangles.pop_back();
    }

  }

  //reconstructed_triangles.resize(0);

}


void LiDARcloud::backfillLeavesAlphaMask( const std::vector<float> leaf_size, const float leaf_aspect_ratio, const float solidfraction, const std::vector<bool> group_filter_flag ){

  if( printmessages ){
    cout << "Backfilling leaves..." << endl;
  }

  srand (time(NULL));

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::minstd_rand0 generator;
  generator.seed(seed);
  std::normal_distribution<float> randn;

  uint Ngroups = reconstructed_triangles.size();

  uint Ncells = getGridCellCount();

  std::vector<std::vector<uint> > group_gridcell;
  group_gridcell.resize(Ncells);

  //Calculate the current alphamask leaf area for each grid cell
  std::vector<float> leaf_area_current;
  leaf_area_current.resize(Ncells);

  int cell;
  int count=0;
  for( uint g=0; g<Ngroups; g++ ){
    if( group_filter_flag.at(g) ){
      if( reconstructed_triangles.at(g).size()>0 ){
	cell = reconstructed_triangles.at(g).front().gridcell;
	leaf_area_current.at(cell) += leaf_size.at(cell)*leaf_size.at(cell)*solidfraction;
	group_gridcell.at(cell).push_back(count);
      }
      count++;
    }
  }

  std::vector<int> deleted_groups;
  int backfill_count = 0;

  //Get the total theoretical leaf area for each grid cell based on LiDAR scan
  for( uint v=0; v<Ncells; v++ ){

    float leaf_area_total = getCellLeafArea(v);

    float reconstruct_frac = (leaf_area_total-leaf_area_current.at(v))/leaf_area_total;

    if( leaf_area_total==0 || reconstructed_alphamasks_size.size()==0 ){//no leaves in gridcell
      if( printmessages ){
	cout << "WARNING: skipping volume #" << v << " because it has no measured leaf area." << endl;
      }
      continue;
    }else if( getTriangleCount()==0 ){
      if( printmessages ){
	cout << "WARNING: skipping volume #" << v << " because it has no triangles." << endl;
      }
      continue;
    }else if( leaf_area_current.at(v)==0 ){//no directly reconstructed leaves in gridcell

      std::vector<SphericalCoord> tri_rots;

      size_t Ntri = 0;
      for( size_t t=0; t<getTriangleCount(); t++ ){
	Triangulation tri = getTriangle(t);
	if( tri.gridcell == v ){
	  helios::vec3 normal = cross( tri.vertex1-tri.vertex0, tri.vertex2-tri.vertex0 );
	  tri_rots.push_back( make_SphericalCoord(cart2sphere(normal).zenith,cart2sphere(normal).azimuth)  );
	}
      }

      while( leaf_area_current.at(v)<leaf_area_total ){

	int randi = round(randu()*(tri_rots.size()-1));

	helios::vec3 cellsize = getCellSize(v);
	helios::vec3 cellcenter = getCellCenter(v);
	float rotation = getCellRotation(v);

	helios::vec3 shift = cellcenter + rotatePoint(helios::make_vec3( (randu()-0.5)*cellsize.x, (randu()-0.5)*cellsize.y, (randu()-0.5)*cellsize.z ),0,rotation);

	reconstructed_alphamasks_center.push_back( shift );
	reconstructed_alphamasks_size.push_back( reconstructed_alphamasks_size.front() );
	reconstructed_alphamasks_rotation.push_back( tri_rots.at(randi) );
	reconstructed_alphamasks_gridcell.push_back( v );
	reconstructed_alphamasks_direct_flag.push_back( 0 );

	leaf_area_current.at(v) += reconstructed_alphamasks_size.back().x*reconstructed_alphamasks_size.back().y*solidfraction;

      }


    }else if( leaf_area_current.at(v)>leaf_area_total ){//too much leaf area in gridcell

      while( leaf_area_current.at(v)>leaf_area_total ){

	int randi = round(randu()*(group_gridcell.at(v).size()-1));

	int group_index = group_gridcell.at(v).at(randi);

	deleted_groups.push_back(group_index);

	leaf_area_current.at(v) -= reconstructed_alphamasks_size.at(group_index).x*reconstructed_alphamasks_size.at(group_index).y*solidfraction;

      }

    }else{ //not enough leaf area in gridcell

      while( leaf_area_current.at(v)<leaf_area_total ){

	int randi = round(randu()*(group_gridcell.at(v).size()-1));

	int group_index = group_gridcell.at(v).at(randi);

	helios::vec3 cellsize = getCellSize(v);
	helios::vec3 cellcenter = getCellCenter(v);
	float rotation = getCellRotation(v);
	helios::vec3 cellanchor = getCellGlobalAnchor(v);

	//helios::vec3 shift = reconstructed_alphamasks_center.at(group_index) + helios::make_vec3( 0.45*(randu()-0.5)*cellsize.x, 0.45*(randu()-0.5)*cellsize.y, 0.45*(randu()-0.5)*cellsize.z ); //uniform shift about group
	helios::vec3 shift = reconstructed_alphamasks_center.at(group_index) + helios::make_vec3( 0.25*randn(generator)*cellsize.x, 0.25*randn(generator)*cellsize.y, 0.25*randn(generator)*cellsize.z ); //Gaussian shift about group
	//helios::vec3 shift = cellcenter + helios::make_vec3( (randu()-0.5)*cellsize.x, (randu()-0.5)*cellsize.y, (randu()-0.5)*cellsize.z ); //uniform shift within voxel
	shift = rotatePointAboutLine(shift,cellanchor,make_vec3(0,0,1),rotation);

	if( group_index>=reconstructed_alphamasks_center.size() ){
	  std::cout << "FAILED: " << group_index << " " << reconstructed_alphamasks_center.size() << " " << randi << std::endl;
	  exit(EXIT_FAILURE);
	}else if( reconstructed_alphamasks_gridcell.at(group_index)!=v ){
	  std::cout << "FAILED: selected leaf group is not from this gridcell" << std::endl;
	  exit(EXIT_FAILURE);
	}

	reconstructed_alphamasks_center.push_back( shift );
	reconstructed_alphamasks_size.push_back( reconstructed_alphamasks_size.at(group_index) );
	reconstructed_alphamasks_rotation.push_back( reconstructed_alphamasks_rotation.at(group_index) );
	reconstructed_alphamasks_gridcell.push_back( v );
	reconstructed_alphamasks_direct_flag.push_back( 0 );

	leaf_area_current.at(v) += reconstructed_alphamasks_size.at(group_index).x*reconstructed_alphamasks_size.at(group_index).y*solidfraction;

	backfill_count++;

      }

    }

  }

  for( uint v=0; v<Ncells; v++ ){

    float leaf_area_total = getCellLeafArea(v);

    float current_area = 0;
    for( uint i=0; i<reconstructed_alphamasks_size.size(); i++ ){
      if( reconstructed_alphamasks_gridcell.at(i)==v ){
  	current_area += reconstructed_alphamasks_size.at(i).x*reconstructed_alphamasks_size.at(i).y*solidfraction;
      }
    }

  }

  if( printmessages ){
    cout << "Backfilled " << backfill_count << " total leaf groups." << endl;
    cout << "Deleted " << deleted_groups.size() << " total leaf groups." << endl;
  }

  for( int i=deleted_groups.size()-1; i>=0; i-- ){
    int group_index = deleted_groups.at(i);
    if( group_index>=0 && group_index<reconstructed_alphamasks_center.size() ){
      //use swap-and-pop method
      std::swap( reconstructed_alphamasks_center.at(group_index), reconstructed_alphamasks_center.back() );
      reconstructed_alphamasks_center.pop_back();
      std::swap( reconstructed_alphamasks_size.at(group_index), reconstructed_alphamasks_size.back() );
      reconstructed_alphamasks_size.pop_back();
      std::swap( reconstructed_alphamasks_rotation.at(group_index), reconstructed_alphamasks_rotation.back() );
      reconstructed_alphamasks_rotation.pop_back();
      std::swap( reconstructed_alphamasks_gridcell.at(group_index), reconstructed_alphamasks_gridcell.back() );
      reconstructed_alphamasks_gridcell.pop_back();
      std::swap( reconstructed_alphamasks_direct_flag.at(group_index), reconstructed_alphamasks_direct_flag.back() );
      reconstructed_alphamasks_direct_flag.pop_back();
    }
  }

  if( printmessages ){
    cout << "done." << endl;
  }

}

void LiDARcloud::calculateLeafAngleCDF( const uint Nbins, std::vector<std::vector<float> > &CDF_theta, std::vector<std::vector<float> > &CDF_phi ){

  uint Ncells = getGridCellCount();

  std::vector<std::vector<float> > PDF_theta, PDF_phi;
  CDF_theta.resize(Ncells);
  PDF_theta.resize(Ncells);
  CDF_phi.resize(Ncells);
  PDF_phi.resize(Ncells);
  for( uint v=0; v<Ncells; v++ ){
    CDF_theta.at(v).resize(Nbins,0.f);
    PDF_theta.at(v).resize(Nbins,0.f);
    CDF_phi.at(v).resize(Nbins,0.f);
    PDF_phi.at(v).resize(Nbins,0.f);
  }
  float db_theta = 0.5*M_PI/Nbins;
  float db_phi = 2.f*M_PI/Nbins;

  //calculate PDF
  for( size_t g=0; g<reconstructed_triangles.size(); g++ ){
    for( size_t t=0; t<reconstructed_triangles.at(g).size(); t++ ){

      float triangle_area = reconstructed_triangles.at(g).at(t).area;

      int gridcell = reconstructed_triangles.at(g).at(t).gridcell;

      helios::vec3 normal = cross( reconstructed_triangles.at(g).at(t).vertex1-reconstructed_triangles.at(g).at(t).vertex0, reconstructed_triangles.at(g).at(t).vertex2-reconstructed_triangles.at(g).at(t).vertex0 );

      helios::SphericalCoord normal_dir = cart2sphere(normal);

      //flipping
      if( normal.z<0 ){
	normal_dir.azimuth += M_PI;
	if( normal_dir.azimuth>2.f*M_PI ){
	  normal_dir.azimuth -= 2.f*M_PI;
	}
	normal_dir.zenith = normal_dir.zenith - 0.5*M_PI;
      }

      int bin_theta = floor(normal_dir.zenith/db_theta);
      if( bin_theta>=Nbins ){
	bin_theta = Nbins-1;
      }

      int bin_phi = floor(normal_dir.azimuth/db_phi);
      if( bin_phi>=Nbins ){
	bin_phi = Nbins-1;
      }

      PDF_theta.at(gridcell).at(bin_theta) += triangle_area;
      PDF_phi.at(gridcell).at(bin_phi) += triangle_area;

    }
  }

  //calculate PDF from CDF
  for( uint v=0; v<Ncells; v++ ){
    for( uint i=0; i<Nbins; i++ ){
      for( uint j=0; j<=i; j++ ){
	CDF_theta.at(v).at(i) += PDF_theta.at(v).at(j);
	CDF_phi.at(v).at(i) += PDF_phi.at(v).at(j);
      }
    }
  }

  char filename[50];
  std::ofstream file_theta, file_phi;
  for( uint v=0; v<Ncells; v++ ){
    sprintf(filename,"../output/PDF_theta%d.txt",v);
    file_theta.open(filename);
    sprintf(filename,"../output/PDF_phi%d.txt",v);
    file_phi.open(filename);
    for( uint i=0; i<Nbins; i++ ){
      file_theta << PDF_theta.at(v).at(i) << std::endl;
      file_phi << PDF_phi.at(v).at(i) << std::endl;
    }
    file_theta.close();
    file_phi.close();
  }

}
