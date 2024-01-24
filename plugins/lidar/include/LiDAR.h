/** \file "LiDAR.h" Primary header file for LiDAR plug-in.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef LIDARPLUGIN
#define LIDARPLUGIN

#include "Context.h"
#include "Visualizer.h"

#include "s_hull_pro.h"

template <class datatype> class HitTable {
public:

  uint Ntheta, Nphi;

  HitTable( void ){
    Ntheta = 0;
    Nphi = 0;
  }
  HitTable( const int nx, const int ny ){
    Ntheta=nx;
    Nphi=ny;
    data.resize(Nphi);
    for( int j=0; j<Nphi; j++ ){
      data.at(j).resize(Ntheta);
    }
  }
  HitTable( const int nx, const int ny, const datatype initval ){
    Ntheta=nx;
    Nphi=ny;
    data.resize(Nphi);
    for( int j=0; j<Nphi; j++ ){
      data.at(j).resize(Ntheta,initval);
    }
  }

  datatype get( const int i, const int j ) const{
    if( i>=0 && i<Ntheta && j>=0 && j<Nphi ){
      return data.at(j).at(i);
    }else{
      std::cerr << "ERROR (hit_map.get): get index out of range. Attempting to get index map at (" << i << "," << j << "), but size of scan is " << Ntheta << " x " << Nphi << "." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  void set( const int i, const int j, const datatype value ){
    if( i>=0 && i<Ntheta && j>=0 && j<Nphi ){
      data.at(j).at(i) = value;
    }else{
      std::cerr << "ERROR (hit_map.set): set index out of range. Attempting to set index map at (" << i << "," << j << "), but size of scan is " << Ntheta << " x " << Nphi << "." << std::endl;
      exit(EXIT_FAILURE);
    }
  }
  void resize( const int nx, const int ny, const datatype initval ){
    Ntheta=nx;
    Nphi=ny;
    data.resize(Nphi);
    for( int j=0; j<Nphi; j++ ){
      data.at(j).resize(Ntheta);
      for( int i=0; i<Ntheta; i++ ){
	data.at(j).at(i) = initval;
      }
    }
  }

private:

  std::vector<std::vector<datatype> > data;

};

struct HitPoint{
  helios::vec3 position;
  helios::SphericalCoord direction;
  helios::int2 row_column;
  helios::RGBcolor color;
  std::map<std::string, double> data;
  int gridcell;
  int scanID;
  HitPoint(void){
    position = helios::make_vec3(0,0,0);
    direction = helios::make_SphericalCoord(0,0);
    row_column = helios::make_int2(0,0);
    color = helios::RGB::red;
    gridcell = -2;
    scanID = -1;
  }
  HitPoint( int __scanID, helios::vec3 __position, helios::SphericalCoord __direction, helios::int2 __row_column, helios::RGBcolor __color, std::map<std::string, double> __data ){
    scanID = __scanID;
    position = __position;
    direction = __direction;
    row_column = __row_column;
    color = __color;
    data = __data;
  }
};

struct Triangulation{
  helios::vec3 vertex0, vertex1, vertex2;
  int ID0, ID1, ID2;
  int scanID;
  int gridcell;
  helios::RGBcolor color;
  float area;
  Triangulation(void){
    vertex0 = helios::make_vec3(0,0,0);
    vertex1 = helios::make_vec3(0,0,0);
    vertex2 = helios::make_vec3(0,0,0);
    ID0 = 0;
    ID1 = 0;
    ID2 = 0;
    scanID = -1;
    gridcell = -2;
    color = helios::RGB::green;
    area = 0;
  }
  Triangulation( int __scanID, helios::vec3 __vertex0, helios::vec3 __vertex1, helios::vec3 __vertex2, int __ID0, int __ID1, int __ID2, helios::RGBcolor __color, int __gridcell ){
    scanID = __scanID;
    vertex0 = __vertex0;
    vertex1 = __vertex1;
    vertex2 = __vertex2;
    ID0 = __ID0;
    ID1 = __ID1;
    ID2 = __ID2;
    gridcell = __gridcell;
    color = __color;

    //calculate area
    helios::vec3 s0 = vertex1-vertex0;
    helios::vec3 s1 = vertex2-vertex0;
    helios::vec3 s2 = vertex2-vertex1;

    float a = s0.magnitude();
    float b = s1.magnitude();
    float c = s2.magnitude();
    float s = 0.5f*(a+b+c);

    area = sqrt( s*(s-a)*(s-b)*(s-c) );
    
  }
};

struct GridCell{
  helios::vec3 center;
  helios::vec3 global_anchor;
  helios::vec3 size;
  helios::vec3 global_size;
  helios::int3 global_ijk;
  helios::int3 global_count;
  float azimuthal_rotation;
  float leaf_area;
  float Gtheta;
  float ground_height;
  float vegetation_height;
  float maximum_height;
  GridCell( helios::vec3 __center, helios::vec3 __global_anchor, helios::vec3 __size, helios::vec3 __global_size, float __azimuthal_rotation, helios::int3 __global_ijk, helios::int3 __global_count ){
    center = __center;
    global_anchor = __global_anchor;
    size = __size;
    global_size = __global_size;
    azimuthal_rotation = __azimuthal_rotation;
    global_ijk = __global_ijk;
    global_count = __global_count;
    leaf_area = 0;
    Gtheta = 0;
    ground_height = 0;
    vegetation_height = 0;
    maximum_height = 0;
  }
};

//! Structure containing metadata for a terrestrial scan
/** A scan is initialized by providing 1) the origin of the scan (see \ref origin), 2) the number of zenithal scan directions (see \ref Ntheta), 3) the range of zenithal scan angles (see \ref thetaMin, \ref thetaMax), 4) the number of azimuthal scan directions (see \ref Nphi), 5) the range of
azimuthal scan angles (see \ref phiMin, \ref phiMax). This creates a grid of Ntheta x Nphi scan points which are all initialized as misses.  Points are set as hits using the addHitPoint() function. There are various functions to query the scan data.
*/
struct ScanMetadata{

  //! Default LiDAR scan data structure
  ScanMetadata();

  //! Create a LiDAR scan data structure
  /**
   * \param[in] "origin" (x,y,z) position of the scanner
   * \param[in] "Ntheta" Number of scan points in the theta (zenithal) direction
   * \param[in] "thetamin" Minimum scan angle in the theta (zenithal) direction in radians
   * \param[in] "thetamax" Maximum scan angle in the theta (zenithal) direction in radians
   * \param[in] "Nphi" Number of scan points in the phi (azimuthal) direction
   * \param[in] "phimin" Minimum scan angle in the phi (azimuthal) direction in radians
   * \param[in] "phimax" Maximum scan angle in the phi (azimuthal) direction in radians
  */
  ScanMetadata( const helios::vec3 &a_origin, uint a_Ntheta, float a_thetaMin, float a_thetaMax, uint a_Nphi, float a_phiMin, float a_phiMax, float a_exitDiameter, float a_beamDivergence, const std::vector<std::string> &a_columnFormat);

  //! File containing hit point data
  std::string data_file;
  
  //! Number of zenithal angles in scan (rows)
  uint Ntheta;                      
  //! Minimum zenithal angle of scan in radians  
  /**
   * \note Zenithal angles range from -pi/2 (downward) to +pi/2 (upward).
   */
  float thetaMin;
  //! Maximum zenithal angle of scan in radians 
  /**
   * \note Zenithal angles range from 0 (upward) to pi (downward).
   */
  float thetaMax;
  
  //! Number of azimuthal angles in scan (columns)
  uint Nphi;      
  //! Minimum azimuthal angle of scan in radians
  /**
   * \note Azimuthal angles start at 0 (x=+,y=0) to 2pi (x=0,y=+) through 2pi.
   */
  float phiMin;
  //! Maximum azimuthal angle of scan in radians
  /**
   * \note Azimuthal angles start at 0 (x=+,y=0) to 2pi (x=0,y=+) through 2pi.
   */
  float phiMax;
  
  //!(x,y,z) coordinate of scanner location
  helios::vec3 origin;

  //! Diameter of laser pulse at exit from the scanner
  /**
   * \note This is not needed for discrete return instruments, and is only used for synthetic scan generation.
   */
  float exitDiameter;

  //! Divergence angle of the laser beam in radians
  /**
   * \note This is not needed for discrete return instruments, and is only used for synthetic scan generation.
   */
  float beamDivergence;

  //! Vector of strings specifying the columns of the scan ASCII file for input/output
  std::vector<std::string> columnFormat;

    //! Convert the (row,column) of hit point in a scan to a direction vector
    /**
     * \param[in] "row" Index of hit point in the theta (zenithal) direction.
     * \param[in] "column" Index of hit point in the phi (azimuthal) direction.
     * \return Spherical vector corresponding to the ray direction for the given hit point.
    */
    helios::SphericalCoord rc2direction(uint row, uint column ) const;

    //! Convert the scan ray direction into (row,column) table index
    /**
     * \param[in] "direction" Spherical vector corresponding to the ray direction for the given hit point.
     * \return (row,column) table index for the given hit point
    */
    helios::int2 direction2rc(const helios::SphericalCoord &direction ) const;
  
};

//! Primary class for terrestrial LiDAR scan
class LiDARcloud{
 private:

  size_t Nhits;

  std::vector<ScanMetadata> scans;

  std::vector<HitPoint> hits;

  std::vector<GridCell> grid_cells;

  std::vector<Triangulation> triangles;

  //!2D map of hits, one value for each (theta,phi) combo of scan. = -1 if no hit, = index if hit - size = (Ntheta)x(Nphi)
  std::vector< HitTable<int> > hit_tables;
    
  //! Flag denoting whether \ref LiDARcloud::calculateHitGridCell[*]() has been called previously.
  bool hitgridcellcomputed;
  
  //! Flag denoting whether triangulation has been performed previously
  bool triangulationcomputed;

  //! Flag denoting whether messages should be printed to screen
  bool printmessages;

  // -------- I/O --------- //

  //! Load point cloud data from a tabular ASCII text file
  /**
    * \param[inout] "scandata" Metadata for point cloud data contained in the ASCII text file.
    * \param[in] "scanID" ID of the scan to which the point cloud data should be added.
    * \return Number of points loaded from the file.
  */
  size_t loadASCIIFile( uint scanID, ScanMetadata &scandata );

  // -------- RECONSTRUCTION --------- //

  // first index: leaf group, second index: triangle #
  std::vector<std::vector<Triangulation> > reconstructed_triangles;

  std::vector<std::vector<Triangulation> > reconstructed_trunk_triangles;

  std::vector<helios::vec3> reconstructed_alphamasks_center;
  std::vector<helios::vec2> reconstructed_alphamasks_size;
  std::vector<helios::SphericalCoord> reconstructed_alphamasks_rotation;
  std::vector<uint> reconstructed_alphamasks_gridcell;
  std::string reconstructed_alphamasks_maskfile;
  std::vector<uint> reconstructed_alphamasks_direct_flag;

  void leafReconstructionFloodfill();

  void backfillLeavesAlphaMask(const std::vector<float> &leaf_size, float leaf_aspect_ratio, float solidfraction, const std::vector<bool> &group_filter_flag );

  void calculateLeafAngleCDF(uint Nbins, std::vector<std::vector<float> > &CDF_theta, std::vector<std::vector<float> > &CDF_phi );
  
  void floodfill( size_t t, std::vector<Triangulation> &cloud_triangles, std::vector<int> &fill_flag, std::vector<std::vector<int> > &nodes, int tag, int depth, int maxdepth );

  void sourcesInsideGridCellGPU();

  //! Perform inversion to estimate LAD
  /**
   * \param[in] "P" Vector of floats where each element is the P value of a given grid cell
   * \param[in] "Gtheta" Vector of floats where each element is the Gtheta value of a given grid cell
   * \param[in] "dr_array"  2D Vector of floats where the first index is the grid cell and the second index is the beam index
   * \param[in] "fillAnalytic" If true the analytic solution using mean dr will be used when the inversion fails. If false, LAD will be set as 999.
   */
  std::vector<float> LAD_inversion(std::vector<float> &P, std::vector<float> &Gtheta, std::vector<std::vector<float>> &dr_array, bool fillAnalytic);

 public:

  //! LiDAR point cloud constructor
   LiDARcloud();

  //! LiDAR point cloud destructor
  ~LiDARcloud();

  //! Self-test (unit test) function
  int selfTest();

  void validateRayDirections();

  //! Disable all print messages to the screen except for fatal error messages
  void disableMessages();

  //! Enable all print messages to the screen
  void enableMessages();

  // ------- SCANS -------- //

  //! Get number of scans in point cloud
  uint getScanCount();

  //! Add a LiDAR scan to the point cloud
  /**
   * \param[in] "newscan" LiDAR scan data structure
   * \return ID for scan that was created
   */
  uint addScan(ScanMetadata &newscan );
    
  //! Specify a scan point as a hit by providing the (x,y,z) coordinates and scan ray direction
  /**
   * \param[in] "scanID" ID of scan hit point to which hit point should be added.
   * \param[in] "xyz" (x,y,z) coordinates of hit point.
   * \param[in] "direction" Spherical coordinate corresponding to the scanner ray direction for the hit point.
   * \note If only the (row,column) scan table coordinates are available, use \ref rc2direction() to convert them to a spherical scan direction coordinate.
  */
  void addHitPoint( uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction );

  //! Specify a scan point as a hit by providing the (x,y,z) coordinates and scan ray direction
  /**
   * \param[in] "scanID" ID of scan hit point to which hit point should be added.
   * \param[in] "xyz" (x,y,z) coordinates of hit point.
   * \param[in] "direction" Spherical coordinate corresponding to the scanner ray direction for the hit point.
   * \param[in] "color" r-g-b color of the hit point
   * \note If only the (row,column) scan table coordinates are available, use \ref rc2direction() to convert them to a spherical scan direction coordinate.
  */
  void addHitPoint( uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction, const helios::RGBcolor &color );

  //! Specify a scan point as a hit by providing the (x,y,z) coordinates  and scan ray direction
  /**
   * \param[in] "scanID" ID of scan hit point to which hit point should be added.
   * \param[in] "xyz" (x,y,z) coordinates of hit point.
   * \param[in] "direction" Spherical coordinate corresponding to the scanner ray direction for the hit point.
   * \param[in] "data" Map data structure containing floating point data values for the hit point.  E.g., "reflectance" could be mapped to a value of 965.2.
  */
  void addHitPoint(uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction, const std::map<std::string, double> &data );
    
  //! Specify a scan point as a hit by providing the (x,y,z) coordinates  and scan ray direction
  /**
   * \param[in] "scanID" ID of scan hit point to which hit point should be added.
   * \param[in] "xyz" (x,y,z) coordinates of hit point.
   * \param[in] "direction" Spherical coordinate corresponding to the scanner ray direction for the hit point.
   * \param[in] "color" r-g-b color of the hit point
   * \param[in] "data" Map data structure containing floating point data values for the hit point.  E.g., "reflectance" could be mapped to a value of 965.2.
  */
  void addHitPoint( uint scanID, const helios::vec3 &xyz, const helios::SphericalCoord &direction, const helios::RGBcolor &color, const std::map<std::string, double> &data );
  
  //! Specify a scan point as a hit by providing the (x,y,z) coordinates and row,column in scan table
  /** 
   * \param[in] "scanID" ID of scan hit point to which hit point should be added.
   * \param[in] "xyz" (x,y,z) coordinates of hit point.
   * \param[in] "row_column" row (theta index) and column (phi index) for point in scan table
   * \param[in] "color" r-g-b color of the hit point
   * \param[in] "data" Map data structure containing floating point data values for the hit point.  E.g., "reflectance" could be mapped to a value of 965.2.
  */
  void addHitPoint( uint scanID, const helios::vec3 &xyz, const helios::int2 &row_column, const helios::RGBcolor &color, const std::map<std::string, double> &data );

  //! Delete a hit point in the scan
  /**
   * \param[in] "index" Index of hit point in the point cloud
  */
  void deleteHitPoint( uint index );

  //! Get the number of hit points in the point cloud
  uint getHitCount() const;
    
  //! Get the (x,y,z) scan origin
  /**
   * \param[in] "scanID" ID of scan.
  */
  helios::vec3 getScanOrigin( uint scanID ) const;

  //! Get the number of scan points in the theta (zenithal) direction
  /**
   * \param[in] "scanID" ID of scan.
  */
  uint getScanSizeTheta( uint scanID ) const;

  //! Get the number of scan points in the phi (azimuthal) direction
  /**
   * \param[in] "scanID" ID of scan.
  */
  uint getScanSizePhi( uint scanID ) const;
  
  //! Get the range of scan directions in the theta (zenithal) direction
  /**
   * \param[in] "scanID" ID of scan.
   * \return vec2.x is the minimum scan zenithal angle, and vec2.y is the maximum scan zenithal angle, both in radians
  */
  helios::vec2 getScanRangeTheta( uint scanID ) const;

  //! Get the range of scan directions in the phi (azimuthal) direction
  /**
   * \param[in] "scanID" ID of scan.
   * \return vec2.x is the minimum scan azimuthal angle, and vec2.y is the maximum scan azimuthal angle, both in radians
  */
  helios::vec2 getScanRangePhi( uint scanID ) const;

  //! Get the diameter of the laser beam at exit from the instrument
  /**
   * \param[in] "scanID" ID of scan.
   * \return Diameter of the beam at exit.
  */
  float getScanBeamExitDiameter( uint scanID ) const;

  //! Get the labels for columns in ASCII input/output file
  /**
   * \param[in] "scanID" ID of scan.
   */
  std::vector<std::string> getScanColumnFormat( uint scanID ) const;

  //! Divergence angle of the laser beam in radians
   /**
    * \param[in] "scanID" ID of scan.
    * \return Divergence angle of the beam.
  */
  float getScanBeamDivergence( uint scanID ) const;

  //! Get (x,y,z) coordinate of hit point by index
  /**
   * \param [in] "index" Hit number
   */
  helios::vec3 getHitXYZ( uint index ) const;

  //! Get ray direction of hit point in the scan based on its index
  /**
   * \param [in] "index" Hit number
   */
  helios::SphericalCoord getHitRaydir( uint index ) const;

  //! Set floating point data value associated with a hit point.
  /**
   * \param[in] "index" Hit number.
   * \param[in] "label" Label of the data value (e.g., "reflectance").
   * \param[in] "value" Value of scalar data.
  */
  double getHitData( uint index, const char* label ) const;

  //! Get floating point data value associated with a hit point.
  /**
   * \param[in] "index" Hit number.
   * \param[in] "label" Label of the data value (e.g., "reflectance").
  */
  void setHitData(uint index, const char* label, double value );

  //! Check if scalar data exists for a hit point
  /**
   * \param[in] "index" Hit number.
   * \param[in] "label" Label of the data value (e.g., "reflectance").
  */
  bool doesHitDataExist( uint index, const char* label ) const;
  
  //! Get color of hit point
  /**
   * \param[in] "index" Hit number
   */
  helios::RGBcolor getHitColor( uint index ) const;

  //! Get the scan with which a hit is associated
  /**
   * \param[in] "index" Hit number
   */
  int getHitScanID( uint index ) const;
  
  //! Get the index of a scan point based on its row and column in the hit table
  /**
   * \param[in] "scanID" ID of scan.
   * \param[in] "row" Row in the 2D scan data table (elevation angle).
   * \param[in] "column" Column in the 2D scan data table (azimuthal angle).
   * \note If the point was not a hit, the function will return `-1'.
  */
  int getHitIndex( uint scanID, uint row, uint column ) const;
  
  //! Get the grid cell in which the hit point resides
  /**
   * \param[in] "index" Hit number
   * \note If the point does not reside in any grid cells, this function returns `-1'.
   * \note Calling this function requires that the function calculateHitGridCell[*]() has been called previously.
   */
  int getHitGridCell( uint index ) const;

  //! Set the grid cell in which the hit point resides
  /**
   * \param[in] "index" Hit number
   * \param[in] "cell" Cell number
  */
  void setHitGridCell(uint index, int cell );

  //! Apply a translation to all points in the point cloud
  /**
   * \param[in] "shift" Distance to translate in x-, y-, and z- direction
   */
  void coordinateShift(const helios::vec3 &shift );

  //! Apply a translation to all points in a given scan
  /**
   * \param[in] "scanID" ID of scan to be shifted
   * \param[in] "shift" Distance to translate in x-, y-, and z- direction
   */
  void coordinateShift( uint scanID, const helios::vec3 &shift );

  //! Rotate all points in the point cloud about the origin
  /**
   * \param[in] "rotation" Spherical rotation angle
   */
  void coordinateRotation( const helios::SphericalCoord &rotation );

  //! Rotate all points in the point cloud about the origin
  /**
   * \param[in] "scanID" ID of scan to be shifted
   * \param[in] "rotation" Spherical rotation angle
   */
  void coordinateRotation( uint scanID, const helios::SphericalCoord &rotation );

  //! Rotate all points in the point cloud about an arbitrary line
  /**
   * \param[in] "rotation" Spherical rotation angle
   * \param[in] "line_base" (x,y,z) coordinate of a point on the line about which points will be rotated
   * \param[in] "line_direction" Unit vector pointing in the direction of the line about which points will be rotated
   */
  void coordinateRotation( float rotation, const helios::vec3 &line_base, const helios::vec3 &line_direction );
    
  //! Get the number of triangles formed by the triangulation
  uint getTriangleCount() const;
    
  //! Get hit point corresponding to first vertex of triangle
  /**
   * \parameter[in] "index" Triangulation index (0 thru Ntriangles-1)
   * \return Hit point index (0 thru Nhits-1)
  */
  Triangulation getTriangle( uint index ) const;

  // ------- FILE I/O --------- //

  //! Read an XML file containing scan information
  /**
   * \param[in] "filename" Path to XML file
   */
  void loadXML( const char* filename );

  //! Read an XML file containing scan information
  /**
   * \param[in] "filename" Path to XML file
   * \param[in] "load_grid_only" if true only the voxel grid defined in the xml file will be loaded, the scans themselves will not be loaded.
   */
  void loadXML( const char* filename, bool load_grid_only );

  //! Export to file the normal vectors (nx,ny,nz) for all triangles formed
  /**
   * \param[in] "filename" Name of file
  */
  void exportTriangleNormals( const char* filename );

  //! Export to file the normal vectors (nx,ny,nz) for triangles formed within a single gridcell
  /**
   * \param[in] "filename" Name of file
   * \param[in] "gridcell" Index of gridcell to get triangles from
  */
  void exportTriangleNormals( const char* filename, int gridcell );

  //! Export to file the area of all triangles formed
  /**
   * \param[in] "filename" Name of file
  */
  void exportTriangleAreas( const char* filename );

  //! Export to file the area of all triangles formed within a single grid cell
  /**
   * \param[in] "filename" Name of file
   * \param[in] "gridcell" Index of gridcell to get triangles from
  */
  void exportTriangleAreas( const char* filename, int gridcell );

  //! Export to file discrete area-weighted inclination angle probability distribution based on the triangulation. Inclination angles are between 0 and 90 degrees. The probability distribution is normalized such that the sine-weighted integral over all angles is 1. The value of each bin is written as a column in the output file; lines correspond to each voxel grid cell.
  /**
    * \param[in] "filename" Name of file
    * \param[in] "Nbins" Number of bins to use for the histogram
  */
  void exportTriangleInclinationDistribution( const char* filename, uint Nbins );

  //! Export to file the leaf area within each grid cell.  Lines of the file correspond to each grid cell
  /**
   * \param[in] "filename" Name of file
  */
  void exportLeafAreas( const char* filename );

  //! Export to file the leaf area density within each grid cell.  Lines of the file correspond to each grid cell
  /**
   * \param[in] "filename" Name of file
  */
  void exportLeafAreaDensities( const char* filename );

  //! Export to file the G(theta) value within each grid cell.  Lines of the file correspond to each grid cell
  /**
   * \param[in] "filename" Name of file
  */
  void exportGtheta( const char* filename );

  //! Export to file all points in the point cloud to an ASCII text file following the column format specified by the <ASCII_format></ASCII_format> tag in the scan XML file 
  /**
   * \param[in] "filename" Name of file
   * \note If there are multiple scans in the point cloud, each scan will be exported to a different file with the scan ID appended to the filename. This is because different scans may have a different column format.
  */
  void exportPointCloud( const char* filename );

  //! Export to file all points from a given scan to an ASCII text file following the column format specified by the <ASCII_format></ASCII_format> tag in the scan XML file 
  /**
   * \param[in] "filename" Name of file
   * \param[in] "scanID" Identifier of scan to be exported
  */
  void exportPointCloud( const char* filename, uint scanID );

    //! Export to file all points from a given scan to PTX file.
    /**
     * \param[in] "filename" Name of file
     * \param[in] "scanID" Identifier of scan to be exported
    */
    void exportPointCloudPTX( const char* filename, uint scanID );

  // ------- VISUALIZER --------- //
  
  //! Add all hit points to the visualizer plug-in, and color them by their r-g-b color
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plugin object.
   * \param[in] "pointsize" Size of scan point in font points.
  */
  void addHitsToVisualizer( Visualizer* visualizer, uint pointsize ) const;
  
  //! Add all hit points to the visualizer plug-in, and color them by a hit scalar data value
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plugin object.
   * \param[in] "pointsize" Size of scan point in font points.
   * \param[in] "color_value" Label for scalar hit data.
  */
  void addHitsToVisualizer( Visualizer* visualizer, uint pointsize, const char* color_value ) const;

  //! Add all grid cells to the visualizer plug-in
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plug-in object.
   */
  void addGridToVisualizer( Visualizer* visualizer ) const;

  //! Add wire frame of the grid to the visualizer plug-in
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plug-in object.
   */
  void addGridWireFrametoVisualizer( Visualizer* visualizer ) const;
  
  //! Add a grid to point cloud instead of reading in from an xml file
  /**
   * \param[in] "center" center of the grid.
   * \param[in] "size" Size of the grid in each dimension.
   * \param[in] "ndiv" number of cells in the grid in each dimension.
   * \param[in] "rotation" horizontal rotation in degrees.
   */
  void addGrid(const helios::vec3 &center, const helios::vec3 &size, const helios::int3 &ndiv, float rotation);

  //! Add all triangles to the visualizer plug-in, and color them by their r-g-b color
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plug-in object.
  */
  void addTrianglesToVisualizer( Visualizer* visualizer ) const;

  //! Add triangles within a given grid cell to the visualizer plug-in, and color them by their r-g-b color
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plugin object.
   * \param[in] "gridcell" Index of grid cell.
  */
  void addTrianglesToVisualizer( Visualizer* visualizer, uint gridcell ) const;

  //! Add reconstructed leaves (triangles or alpha masks) to the visualizer plug-in
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plugin object.
   */
  void addLeafReconstructionToVisualizer( Visualizer* visualizer ) const;

  //! Add trunk reconstruction to the visualizer plug-in.  Colors reconstructed triangles by hit point color.
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plugin object.
   */
  void addTrunkReconstructionToVisualizer( Visualizer* visualizer ) const;

  //! Add trunk reconstruction to the visualizer plug-in
  /**
   * \param[in] "visualizer" Pointer to the Visualizer plugin object.
   * \param[in] "trunk_color" r-g-b color of trunk.
   */
  void addTrunkReconstructionToVisualizer( Visualizer* visualizer, const helios::RGBcolor &trunk_color ) const;
  
  //! Add reconstructed leaves (texture-masked patches) to the Context
  /**
   * \param[in] "context" Pointer to the Helios context
   * \note This function creates the following primitive data for each patch 1) ``gridCell" which indicates the index of the gridcell that contains the patch, 2) ``directFlag" which equals 1 if the leaf was part of the direct reconstruction, and 0 if the leaf was backfilled.
   */
  std::vector<uint> addLeafReconstructionToContext( helios::Context* context ) const;

  //! Add reconstructed leaves (texture-masked patches) to the Context with leaves divided into sub-patches (tiled)
  /**
   * \param[in] "context" Pointer to the Helios context
   * \param[in] "subpatches" Number of leaf sub-patches (tiles) in the x- and y- directions.
   * \note This function creates the following primitive data for each patch 1) ``gridCell" which indicates the index of the gridcell that contains the patch, 2) ``directFlag" which equals 1 if the leaf was part of the direct reconstruction, and 0 if the leaf was backfilled.
   */
  std::vector<uint> addLeafReconstructionToContext( helios::Context* context, const helios::int2 &subpatches ) const;

  //! Add triangle groups used in the direct reconstruction to the Context
  /**
   * \param[in] "context" Pointer to the Helios context.
   * \note This function creates primitive data called ``leafGroup" which provides an identifier for each triangle based on the fill group it is in.
  */
  std::vector<uint> addReconstructedTriangleGroupsToContext( helios::Context* context ) const;

  //! Add reconstructed trunk triangles to the Context
  /**
   * \param[in] "context" Pointer to the Helios context
   */
  std::vector<uint> addTrunkReconstructionToContext( helios::Context* context ) const;
  
  //! Form an axis-aligned bounding box for all hit points in the point cloud
  /**
   * \param[out] "boxmin" Coordinates of the bounding box vertex in the (-x,-y,-z) direction.
   * \param[out] "boxmax" Coordinates of the bounding box vertex in the (+x,+y,+z) direction.
  */
  void getHitBoundingBox( helios::vec3& boxmin, helios::vec3& boxmax ) const;

  //! Form an axis-aligned bounding box for all grid cells in the point cloud
  /**
   * \param[out] "boxmin" Coordinates of the bounding box vertex in the (-x,-y,-z) direction.
   * \param[out] "boxmax" Coordinates of the bounding box vertex in the (+x,+y,+z) direction.
  */
  void getGridBoundingBox( helios::vec3& boxmin, helios::vec3& boxmax ) const;

  // --------- POINT FILTERING ----------- //

  //! Filter scan by imposing a maximum distance from the scanner
  /**
   * \param[in] "maxdistance" Maximum hit point distance from scanner
   */
  void distanceFilter( float maxdistance );
  
  //! overloaded version of xyzFilter that defaults to deleting points outside the provided bounding box
  /**
   * \param[in] "xmin" minimum x coordinate of bounding box
   * \param[in] "xmax" maximum x coordinate of bounding box
   * \param[in] "ymin" minimum y coordinate of bounding box
   * \param[in] "ymax" maximum y coordinate of bounding box
   * \param[in] "zmin" minimum z coordinate of bounding box
   * \param[in] "zmax" maximum z coordinate of bounding box
   * \note points outside the provided bounding box are deleted by default
   */
  void xyzFilter( float xmin, float xmax, float ymin, float ymax, float zmin, float zmax );

  //! Filter scan with a bounding box
  /**
   * \param[in] "xmin" minimum x coordinate of bounding box
   * \param[in] "xmax" maximum x coordinate of bounding box
   * \param[in] "ymin" minimum y coordinate of bounding box
   * \param[in] "ymax" maximum y coordinate of bounding box
   * \param[in] "zmin" minimum z coordinate of bounding box
   * \param[in] "zmax" maximum z coordinate of bounding box
   * \param[in] "deleteOutside" if true, deletes points outside the bounding box, if false deletes points inside the bounding box
   * \note points outside the provided bounding box are deleted
   */
  void xyzFilter( float xmin, float xmax, float ymin, float ymax, float zmin, float zmax, bool deleteOutside );


  //! Filter scan by imposing a minimum reflectance value
  /**
   * \param[in] "minreflectance" Miniimum hit point reflectance value
   * \note If `reflectance' data was not provided for a hit point when calling \ref Scan::addHitPoint(), the point will not be filtered.
  */
  void reflectanceFilter( float minreflectance );

  //! Filter hit points based on a scalar field given by a column in the ASCII data
  /**
   * \param[in] "scalar_field" Name of a scalar field defined in the ASCII point cloud data (e.g., "reflectance")
   * \param[in] "threshold" Value for filter threshold
   * \param[in] "comparator" Points will be filtered if "scalar (comparator) threshold", where (comparator) is one of ">", "<", or "="
   * \note As an example, imagine we wanted to remove all hit points where the reflectance is less than -10. In this case we would call scalarFilter( "reflectance", -10, "<" );
  */
  void scalarFilter( const char* scalar_field, float threshold, const char* comparator );

  //! Filter full-waveform data according to the maximum scalar value along each pulse. Any scalar value can be used, provided it is a field in the hit point data file. The resulting point cloud will have only one hit point per laser pulse.
  /**
   * \param[in] "scalar" Name of hit point scalar data in the hit data file.
   * \note This function is only applicable for full-waveform data and requires that the scalar field "timestamp" is provided in the ASCII hit point data file.
  */
  void maxPulseFilter( const char* scalar );

  //! Filter full-waveform data according to the minimum scalar value along each pulse. Any scalar value can be used, provided it is a field in the hit point data file. The resulting point cloud will have only one hit point per laser pulse.
  /**
   * \param[in] "scalar" Name of hit point scalar data in the ASCII hit data file.
   * \note This function is only applicable for full-waveform data and requires that the scalar field "timestamp" is provided in the hit point data file.
  */
  void minPulseFilter( const char* scalar );

  //! Filter full-waveform data to include only the first hit per laser pulse. The resulting point cloud will have only one hit point per laser pulse (first hits).
  /**
   * \note This function is only applicable for full-waveform data and requires that the scalar field "target_index" is provided in the hit point data file. The "target_index" values can start at 0 or 1 for first hits as long as it is consistent throughout the point cloud.
  */
  void firstHitFilter();

  //! Filter full-waveform data to include only the last hit per laser pulse. The resulting point cloud will have only one hit point per laser pulse (last hits).
  /**
   * \note This function is only applicable for full-waveform data and requires that the scalar fields "target_index" and "target_count" are provided in the hit point data file. The "target_index" values can start at 0 or 1 for first hits as long as it is consistent throughout the point cloud.
  */
  void lastHitFilter();

  // ------- TRIANGULATION --------- //
  
  //! Perform triangulation on all hit points in point cloud
  /**
   * \param[in] "Lmax" Maximum allowable length of triangle sides.
   * \param[in] "max_aspect_ratio" Maximum allowable aspect ratio of triangles.
  */
  void triangulateHitPoints( float Lmax, float max_aspect_ratio );

  //ERK
  //! Perform triangulation on hit points in point cloud that meet some filtering criteria based on scalar data
  /**
   * \param[in] "Lmax" Maximum allowable length of triangle sides.
   * \param[in] "max_aspect_ratio" Maximum allowable aspect ratio of triangles.
   * * \param[in] "scalar_field" Name of a scalar field defined in the ASCII point cloud data (e.g., "deviation")
   * \param[in] "threshold" Value for filter threshold
   * \param[in] "comparator" Points will not be used in triangulation if "scalar (comparator) threshold", where (comparator) is one of ">", "<", or "="
   * \note As an example, imagine we wanted to remove all hit points where the deviation is greater than 15 for the purposes of the triangulation. In this case we would call triangulateHitPoints(Lmax, max_aspect_ratio, "deviation", 15, ">" );
   */
  void triangulateHitPoints( float Lmax, float max_aspect_ratio, const char* scalar_field, float threshold, const char* comparator );
  
  
  //! Add triangle geometry to Helios context
  /**
   * \parameter[in] "context" Pointer to Helios context
   */
  void addTrianglesToContext( helios::Context* context ) const;

  // -------- GRID ----------- //

  //! Get the number of cells in the grid
  uint getGridCellCount() const;

  //! Add a cell to the grid
  /**
   * \param [in] "center" (x,y,z) coordinate of grid center.
   * \param [in] "size" size of the grid cell in the x,y,z directions.
   * \param [in] "rotation" rotation angle (in radians) of the grid cell about the z-axis.
  */
  void addGridCell( const helios::vec3 &center, const helios::vec3 &size, float rotation );

  //! Add a cell to the grid, where the cell is part of a larger global rectangular grid
  /**
   * \param [in] "center" (x,y,z) coordinate of grid center
   * \param [in] "global_anchor" (x,y,z) coordinate of grid global anchor, i.e., this is the 'center' coordinate entered in the xml file.  If grid Nx=Ny=Nz=1, global_anchor=center
   * \param [in] "size" size of the grid cell in the x,y,z directions
   * \param [in] "global_size" size of the global grid in the x,y,z directions
   * \param [in] "rotation" rotation angle (in radians) of the grid cell about the z-axis
   * \param [in] "global_ijk" index within the global grid in the x,y,z directions
   * \param [in] "global_count" total number of cells in global grid in the x,y,z directions
  */
  void addGridCell( const helios::vec3 &center, const helios::vec3 &global_anchor, const helios::vec3 &size, const helios::vec3 &global_size, float rotation, const helios::int3 &global_ijk, const helios::int3 &global_count );

  //! Get the (x,y,z) coordinate of a grid cell by its index
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  helios::vec3 getCellCenter( uint index ) const;

  //! Get the (x,y,z) coordinate of a grid global anchor by its index
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  helios::vec3 getCellGlobalAnchor( uint index ) const;

  //! Get the size of a grid cell by its index
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, an
d the last cell's index is Ncells-1.
   */
  helios::vec3 getCellSize( uint index ) const;

  //! Get the size of a grid cell by its index
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  float getCellRotation(uint index ) const;

  //! Determine the grid cell in which each hit point resides for the whole point cloud - GPU accelerated version */
  /**
   * \note This function does not return a value, rather, it set the Scan variable `hit_vol' which is queried by the function `Scan::getHitGridCell()'.
   */
  void calculateHitGridCellGPU();

  // ------- SYNTHETIC SCAN ------ //

  //! Run a discrete return synthetic LiDAR scan based on scan parameters given in an XML file (returns only one laser hit per pulse)
  /**
   * \param[in] "context" Pointer to the Helios context
  */
  void syntheticScan( helios::Context* context );

  //! Run a discrete return synthetic LiDAR scan based on scan parameters given in an XML file (returns only one laser hit per pulse)
  /**
   * \param[in] "context" Pointer to the Helios context.
   * \param[in] "scan_grid_only" If true, only record hit points for rays that intersect the voxel grid.
   * \param[in] "record_misses" If true, "miss" points (i.e., beam did not hit any primitives) are recorded in the scan.
   * \note Calling syntheticScan() with scan_grid_only=true can save substantial memory for contexts with large domains.
  */
  void syntheticScan( helios::Context* context, bool scan_grid_only, bool record_misses );

  //! Run a full-waveform synthetic LiDAR scan based on scan parameters given in an XML file (returns multiple laser hits per pulse)
  /**
   * \param[in] "context" Pointer to the Helios context.
   * \param[in] "xml_file" Path to an XML file with LiDAR scan and grid information.
   * \param[in] "rays_per_pulse" Number of ray launches per laser pulse direction.
   * \param[in] "pulse_distance_threshold" Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
   * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a discrete return synthetic scan.
  */
  void syntheticScan( helios::Context* context, int rays_per_pulse, float pulse_distance_threshold );

  //! Run a full-waveform synthetic LiDAR scan based on scan parameters given in an XML file (returns multiple laser hits per pulse)
  /**
   * \param[in] "context" Pointer to the Helios context.
   * \param[in] "rays_per_pulse" Number of ray launches per laser pulse direction.
   * \param[in] "pulse_distance_threshold" Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
   * \param[in] "scan_grid_only" If true, only considers context geometry within the scan grid. scan_grid_only=true can save substantial memory for contexts with large domains.
   * \param[in] "record_misses" If true, "miss" points (i.e., beam did not hit any primitives) are recorded in the scan.
   * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a discrete return synthetic scan.
  */
  void syntheticScan( helios::Context* context, int rays_per_pulse, float pulse_distance_threshold, bool scan_grid_only, bool record_misses );

  //! Calculate the surface area of all primitives in the context
  /**
   * \param[in] "context" Pointer to the Helios context
  */
  std::vector<float> calculateSyntheticLeafArea( helios::Context* context );

  //! Calculate the G(theta) of all primitives in the context
  /**
   * \param[in] "context" Pointer to the Helios context
  */
  std::vector<float> calculateSyntheticGtheta( helios::Context* context );

  // -------- LEAF AREA -------- //

  //! Set the leaf area of a grid cell in m^2
  /**
   * \param[in] "area" Leaf area in cell in m^2.
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  void setCellLeafArea( float area, uint index );

  //! Get the leaf area of a grid cell in m^2
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  float getCellLeafArea( uint index ) const;

  //! Get the leaf area density of a grid cell in 1/m
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  float getCellLeafAreaDensity( uint index ) const;

  //! Set the average G(theta) value of a grid cell
  /**
   * \param[in] "Gtheta" G(theta) in cell.
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  void setCellGtheta( float Gtheta, uint index );

  //! Get the G(theta) of a grid cell
  /**
   * \param [in] "index" Index of a grid cell.  Note: the index of a grid cell is given by the order in which it was added to the grid. E.g., the first cell's index is 0, and the last cell's index is Ncells-1.
   */
  float getCellGtheta( uint index ) const;

    //! For scans that are missing points (e.g., sky points), this function will attempt to fill in missing points for all scans. This increases the accuracy of LAD calculations because it makes sure all pulses are accounted for.
    /**
     * \return (x,y,z) of missing points added to the scan from gapfilling
     */
    std::vector<helios::vec3> gapfillMisses();

  //! For scans that are missing points (e.g., sky points), this function will attempt to fill in missing points. This increases the accuracy of LAD calculations because it makes sure all pulses are accounted for.
  /**
   * \param[in] "scanID" ID of scan to gapfill
   * \return (x,y,z) of missing points added to the scan from gapfilling
   */
  std::vector<helios::vec3> gapfillMisses( uint scanID );
  
  //! For scans that are missing points (e.g., sky points), this function will attempt to fill in missing points. This increases the accuracy of LAD calculations because it makes sure all pulses are accounted for.
  /**
   * \param[in] "scanID" ID of scan to gapfill
   * \param[in] "gapfill_grid_only" if true, missing points are gapfilled only within the axis-aligned bounding box of the voxel grid. If false missing points are gap filled across the range of phi and theta values specified in the scan xml file.
   * \param[in] "add_flags" if true, gapfillMisses_code is added as hitpoint data. 0 = original points, 1 = gapfilled, 2 = extrapolated at downward edge, 3 = extrapolated at upward edge 
   * \return (x,y,z) of missing points added to the scan from gapfilling
   */
  std::vector<helios::vec3> gapfillMisses( uint scanID, const bool gapfill_grid_only, const bool add_flags );
  

  //! Calculate the leaf area for each grid volume
  void calculateLeafAreaGPU();

  //! Calculate the leaf area for each grid volume
  /**
   * \param [in] "min_voxel_hits" Minimum number of allowable LiDAR hits per voxel. If the total number of hits in a voxel is less than min_voxel_hits, the calculated leaf area will be set to zero.
   * \note Currently, this version assumes all data is discrete-return. The function calculateLeafAreaGPU_testing() deals with waveform data, but may not be working correctly. In the next version, these two functions will be combined.
   */
  void calculateLeafAreaGPU( int min_voxel_hits );

  //! Calculate the leaf area for each grid volume
  /**
   * \param [in] "min_voxel_hits" Minimum number of allowable LiDAR hits per voxel. If the total number of hits in a voxel is less than min_voxel_hits, the calculated leaf area will be set to zero.
   */
  void calculateLeafAreaGPU_testing( int min_voxel_hits );

 //! Calculate the leaf area for each grid volume in a synthetic scan using several different method for estimating P 
  /**
   * \param [in] "context" Pointer to the Helios context.
   * \param [in] "beamoutput" if true writes detailed data about each beam to ../beamoutput/beam_data_s_[scan index]_c_[grid cell index].txt.
   * \param [in] "fillAnalytic" if true, when the iterative LAD inversion fails, the analytic solution using mean dr will be substituted. If false LAD is set to 999.
   * \note writes voxel level data to ../voxeloutput/voxeloutput.txt
  */
  void calculateLeafAreaGPU_synthetic( helios::Context* context,  bool beamoutput, bool fillAnalytic );

  //! Calculate the leaf area for each grid volume using equal weighting method
  /**
   * \param [in] "beamoutput" if true writes detailed data about each beam to ../beamoutput/beam_data_s_[scan index]_c_[grid cell index].txt.
   * \param [in] "fillAnalytic" if true, when the iterative LAD inversion fails, the analytic solution using mean dr will be substituted. If false LAD is set to 999.
   * \note writes voxel level data to ../voxeloutput/voxeloutput.txt
   */
  void calculateLeafAreaGPU_equal_weighting( bool beamoutput, bool fillAnalytic );
  
  //! Calculate the leaf area for each grid volume using equal weighting method
  /**
   * \param [in] "beamoutput" if true writes detailed data about each beam to ../beamoutput/beam_data_s_[scan index]_c_[grid cell index].txt.
   * \param [in] "fillAnalytic" if true, when the iterative LAD inversion fails, the analytic solution using mean dr will be substituted. If false LAD is set to 999.
   * \param [in] "constant_G" A separate LAD inversion will be performed for each element of this vector, setting the value of G in all voxels to the value given in this vector.
   * \note writes voxel level data to ../voxeloutput/voxeloutput.txt
   */
  void calculateLeafAreaGPU_equal_weighting( bool beamoutput, bool fillAnalytic, std::vector<float> constant_G );
  
  
  // -------- RECONSTRUCTION --------- //

  //! Perform a leaf reconstruction based on texture-masked Patches within each gridcell.  The reconstruction produces Patches for each reconstructed leaf surface, with leaf size automatically estimated algorithmically.  
  /**
   * \param[in] "minimum_leaf_group_area" Minimum allowable area of leaf triangular fill groups. Leaf fill groups with total areas less than minimum_leaf_group_area are not considered in the reconstruction.
   * \param[in] "maximum_leaf_group_area" Maximum area of leaf triangular fill groups. Leaf fill groups with total areas greater than maximum_leaf_group_area are not considered in the reconstruction.
   * \param[in] "leaf_aspect_ratio" Ratio of length of leaf along midrib to with of leaf perpendicular to leaf midrib.  This will generally be the length/width of leaf mask.
   * \param[in] "mask_file" Path to PNG image file to be used with Alpha Mask.
  */
  void leafReconstructionAlphaMask( float minimum_leaf_group_area, float maximum_leaf_group_area, float leaf_aspect_ratio, const char* mask_file );

  //! Perform a leaf reconstruction based on texture-masked Patches within each gridcell.  The reconstruction produces Patches for each reconstructed leaf surface, with leaf size set to a constant value.
  /**
   * \param[in] "minimum_leaf_group_area" Minimum allowable area of leaf triangular fill groups. Leaf fill groups with total areas less than minimum_leaf_group_area are not considered in the reconstruction.
   * \param[in] "maximum_leaf_group_area" Maximum area of leaf triangular fill groups. Leaf fill groups with total areas greater than maximum_leaf_group_area are not considered in the reconstruction.
   * \param[in] "leaf_aspect_ratio" Ratio of length of leaf along midrib to with of leaf perpendicular to leaf midrib.  This will generally be the length/width of leaf mask.
   * \param[in] "leaf_length_constant" Constant length of all reconstructed leaves.
   * \param[in] "mask_file" Path to PNG image file to be used with Alpha Mask.
  */
  void leafReconstructionAlphaMask( float minimum_leaf_group_area, float maximum_leaf_group_area, float leaf_aspect_ratio, float leaf_length_constant, const char* mask_file );

  //! Reconstruct the trunk of the tree. In order to do this, you must specify the center and size of a rectangular box that encompasses the tree trunk. This routine will then try to find the largest continuous triangle group, which is assumed to correspond to the trunk.
  /**
   * \param[in] "box_center" (x,y,z) coordinates of the center of a rectangular box that encompasses the tree trunk.
   * \param[in] "box_size" Dimension of the trunk box in the x-, y-, and z- directions.
   * \param[in] "Lmax" maximum dimension of triangles (see also triangulateHitPoints()).
   * \param[in] "max_aspect_ratio" Maximum allowable aspect ratio of triangles (see also triangulateHitPoints())
  */
  void trunkReconstruction( const helios::vec3 &box_center, const helios::vec3 &box_size, float Lmax, float max_aspect_ratio );
  
  //! Delete hitpoints that do not pass through / intersect the voxel grid
  /**
   * \param[in] "source" the scan index
   */
  void cropBeamsToGridAngleRange(uint source);
  
  //! find the indices of the peaks of a vector of floats
  /**
   * \param[in] "signal" the signal we want to detect peaks in
   */
  std::vector<uint> peakFinder(std::vector<float> signal);

};

bool sortcol0( const std::vector<double>& v0, const std::vector<double>& v1 );

bool sortcol1( const std::vector<double>& v0, const std::vector<double>& v1 );

#endif
