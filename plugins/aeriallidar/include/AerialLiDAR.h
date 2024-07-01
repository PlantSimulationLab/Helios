/** \file "AerialLiDAR.h" Header file for Aerial LiDAR plug-in dealing with aerial scans.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HELIOS_AERIALLIDAR
#define HELIOS_AERIALLIDAR

#include "Context.h"
#include "Visualizer.h"

//! Structure containing metadata for an aerial scan
/** A scan is initialized by providing 1) the origin of the scan (see \ref origin), 2) the number of zenithal scan directions (see \ref Ntheta), 3) the range of zenithal scan angles (see \ref thetaMin, \ref thetaMax), 4) the number of azimuthal scan directions (see \ref Nphi), 5) the range of azimuthal scan angles (see \ref phiMin, \ref phiMax). This creates a grid of Ntheta x Nphi scan points which are all initialized as misses.  Points are set as hits using the addHitPoint() function. There are various functions to query the scan data.
*/
struct AerialScanMetadata{

  //! Create an aerial LiDAR scan data structure
  /**
   * \param[in] center (x,y,z) position of scan surface center
   * \param[in] extent (x,y) size/extent of scan surface
   * \param[in] coneangle Width of scan cone in degrees
  */
  AerialScanMetadata( const helios::vec3 __center, const helios::vec2 __extent, const float __coneangle, const float __scandensity, const float __exitDiameter, const float __beamDivergence );

  //! Number of laser pulses in scan
  std::size_t Nrays;  
  
  //! (x,y,z) position of scan surface center
  helios::vec3 center;

  //! (x,y) size/extent of scan surface
  helios::vec2 extent;

  //! Width of scan cone in degrees
  float coneangle;

  //! Scan density in points/m^2
  float scandensity;

  //! Diameter of laser pulse at exit from the scanner
  /** \note This is not needed for discrete return instruments. */
  float exitDiameter;

  //! Divergence angle of the laser beam in radians
  /**
   * \note This is not needed for discrete return instruments.
   */
  float beamDivergence;
  
};

//! Structure containing metadata for an aerial hit point
struct AerialHitPoint{
  helios::vec3 position;
  helios::SphericalCoord direction;
  helios::RGBcolor color;
  std::map<std::string, float> data;
  helios::int3 gridcell;
  int scanID;
  AerialHitPoint(void){
    position = helios::make_vec3(0,0,0);
    direction = helios::make_SphericalCoord(0,0);
    color = helios::RGB::red;
    gridcell = helios::make_int3(-2,-2,-2);
    scanID = -1;
  }
  AerialHitPoint( int __scanID, helios::vec3 __position, helios::SphericalCoord __direction, helios::int2 __row_column, helios::RGBcolor __color, std::map<std::string, float> __data ){
    scanID = __scanID;
    position = __position;
    direction = __direction;
    color = __color;
    data = __data;
  }
};

//! Primary class for aerial LiDAR scan
class AerialLiDARcloud{
 private:

  //! Use RANSAC algorithm to separate a set of hit points into outliers and inliers based on proximity to a best-fit plane
  /**
    * \param[in] maxIter Maximum number of iterations to find best fit plane.
    * \param[in] threshDist Maximum distance from fitted plane to be considered an inlier.
    * \param[in] inlierRatio Minimum fraction of total points that must be inliers to consider the fitted plane valid.
    * \param[in] hits Vector of (x,y,z) positions for hit points.
    * \param[out] inliers Vector of flags denoting points as inliers or outliers (value=false means an outlier, value=true means an inlier).
    * \return Best fit plane to the set of inliers. The vec4 contains the four coefficients of the fitted plane equation Ax+By+Cz+D=0. (x=A, y=B, z=C, w=D)
  */
  helios::vec4 RANSAC( const int maxIter, const float threshDist, const float inlierRatio, const std::vector<helios::vec3>& hits, std::vector<bool>& inliers );

  std::vector<AerialScanMetadata> scans;

  std::vector<AerialHitPoint> hits;
  
  //! Flag denoting whether messages should be printed to screen
  bool printmessages;

  // --- grid --- //

  //! Flag denoting whether LiDARcloud::calculateHitGridCell[*]() has been called previously.
  bool hitgridcellcomputed;

  helios::vec3 gridcenter;

  helios::vec3 gridextent;
  
  float gridrotation;

  helios::int3 gridresolution;

  std::vector<std::vector<std::vector<float> > > leaf_area;

  std::vector<std::vector<std::vector<float> > > r_bar;

  std::vector<std::vector<std::vector<helios::int2> > > P_counts;

  bool groundheightcomputed;

  std::vector<std::vector<float> > ground_height;

  std::vector<std::vector<float> > vegetation_height;

  std::vector<std::vector<float> > maxhit_height;

  std::vector<std::vector<float> > cover_fraction;

  helios::vec2 rotatePoint2D( const helios::vec2 point, const float theta ) const;
  
 public:

  //! Aerial LiDAR point cloud constructor
  AerialLiDARcloud( void );

  //! Aerial LiDAR point cloud destructor
  ~AerialLiDARcloud( void );

  //! Self-test (unit test) function
  int selfTest( void );

  //! Disable all print messages to the screen except for fatal error messages
  void disableMessages( void );

  //! Enable all print messages to the screen
  void enableMessages( void );

  // ------- SCANS -------- //

  //! Get number of scans in point cloud
  uint getScanCount( void );

  //! Add a LiDAR scan to the point cloud
  /**
   * \param[in] newscan LiDAR scan data structure
   */
  void addScan( const AerialScanMetadata newscan );

  //! Specify a scan point as a hit by providing the (x,y,z) coordinates of the origin and hit point
  /** 
    * \param[in] scanID ID of scan hit point to which hit point should be added.
    * \param[in] hit_xyz (x,y,z) coordinates of hit point.
    * \param[in] ray_origin (x,y,z) coordinates of ray origin
  */
  void addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::vec3 ray_origin );
    
  //! Specify a scan point as a hit by providing the (x,y,z) coordinates of the hit and scan ray direction
  /** 
    * \param[in] scanID ID of scan hit point to which hit point should be added.
    * \param[in] hit_xyz (x,y,z) coordinates of hit point.
    * \param[in] direction Spherical coordinate cooresponding to the scanner ray direction for the hit point.
  */
  void addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction );

  //! Specify a scan point as a hit by providing the (x,y,z) coordinates of the hit and scan ray direction
  /** 
    * \param[in] scanID ID of scan hit point to which hit point should be added.
    * \param[in] hit_xyz (x,y,z) coordinates of hit point.
    * \param[in] direction Spherical coordinate cooresponding to the scanner ray direction for the hit point.
    * \param[in] color r-g-b color of the hit point
    * \note If only the (row,column) scan table coordinates are available, use \ref rc2direction() to convert them to a spherical scan direction coordinate.
  */
  void addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction, const helios::RGBcolor color );

  //! Specify a scan point as a hit by providing the (x,y,z) coordinates of the hit and scan ray direction
  /** 
    * \param[in] scanID ID of scan hit point to which hit point should be added.
    * \param[in] hit_xyz (x,y,z) coordinates of hit point.
    * \param[in] direction Spherical coordinate cooresponding to the scanner ray direction for the hit point.
    * \param[in] data Map data structure containing floating point data values for the hit point.  E.g., "reflectance" could be mapped to a value of 965.2.
  */
  void addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction, const std::map<std::string, float> data );
    
  //! Specify a scan point as a hit by providing the (x,y,z) coordinates of the hit and scan ray direction
  /** 
    * \param[in] scanID ID of scan hit point to which hit point should be added.
    * \param[in] hit_xyz (x,y,z) coordinates of hit point.
    * \param[in] direction Spherical coordinate cooresponding to the scanner ray direction for the hit point.
    * \param[in] color r-g-b color of the hit point
    * \param[in] data Map data structure containing floating point data values for the hit point.  E.g., "reflectance" could be mapped to a value of 965.2.
  */
  void addHitPoint( const uint scanID, const helios::vec3 hit_xyz, const helios::SphericalCoord direction, const helios::RGBcolor color, const std::map<std::string, float> data );

  //! Delete a hit point in the scan
  /** 
    * \param[in] index Index of hit point in the point cloud
  */
  void deleteHitPoint( const uint index );

  //! Get the number of hit points in the point cloud
  uint getHitCount( void ) const;
    
  //! Get the (x,y,z) of scan surface center
  /** 
    * \param[in] scanID ID of scan.
  */
  helios::vec3 getScanCenter( const uint scanID ) const;

  //! Get the (x,y) extent of scan surface
  /** 
    * \param[in] scanID ID of scan.
  */
  helios::vec2 getScanExtent( const uint scanID ) const;

  //! Get the scan cone angle in degrees
  /** 
    * \param[in] scanID ID of scan.
  */
  float getScanConeAngle( const uint scanID ) const;

  //! Get the scan point density in points/m^2
  float getScanDensity( const uint scanID ) const;

  //! Get the diameter of the laser beam at exit from the instrument
  /**
   * \param[in] scanID ID of scan.
   * \return Diameter of the beam at exit.
  */
  float getScanBeamExitDiameter( const uint scanID ) const;

  //! Divergence angle of the laser beam in radians
   /**
    * \param[in] scanID ID of scan.
    * \return Divergence angle of the beam.
  */
  float getScanBeamDivergence( const uint scanID ) const;

  //! Get (x,y,z) coordinate of hit point by index
  /**
   * \param [in] index Hit number
   */
  helios::vec3 getHitXYZ( uint index ) const;

  //! Get ray direction of hit point in the scan based on its index
  /**
   * \param [in] index Hit number
   */
  helios::SphericalCoord getHitRaydir( const uint index ) const;

  //! Get floating point data value associated with a hit point.
  /**
   * \param[in] index Hit number.
    * \param[in] label Label of the data value (e.g., "reflectance").
    * \param[in] value Value of scalar data.
  */
  float getHitData( const uint index, const char* label ) const;

  //! Set floating point data value associated with a hit point.
  /**
   * \param[in] index Hit number.
    * \param[in] label Label of the data value (e.g., "reflectance").
  */
  void setHitData( const uint index, const char* label, const float value );

  //! Check if scalar data exists for a hit point
  /**
   * \param[in] index Hit number.
    * \param[in] label Label of the data value (e.g., "reflectance").
  */
  bool doesHitDataExist( const uint index, const char* label ) const;
  
  //! Get color of hit point
  /** \param[in] index Hit number */
  helios::RGBcolor getHitColor( const uint index ) const;

  //! Get the scan with which a hit is associated
  /**
   * \param[in] index Hit number
   */
  int getHitScanID( const uint index ) const;
  
  //! Get the grid cell in which the hit point resides
  /**
   * \param[in] index Hit number
   * \note If the point does not reside in any grid cells, this function returns `(-1,-1,-1)'.
   * \note Calling this function requires that the function calculateHitGridCell[*]() has been called previously.
   */
  helios::int3 getHitGridCell( const uint index ) const;

  //! Set the grid cell in which the hit point resides
  /**
   * \param[in] index Hit number
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
  */
  void setHitGridCell( const uint index, const helios::int3 ijk );

  void coordinateShift( const helios::vec3 shift );
  
  // ------- FILE I/O --------- //

  //! Read an XML file containing scan information
  /** \param[in] filename Path to XML file
   */
  void loadXML( const char* filename );
  
  //! Read all XML files currently loaded into the Helios context
  void readContextXML( void );

  //! Export to file all points in the point cloud
  /** 
    * \param[in] filename Name of file
  */
  void exportPointCloud( const char* filename );

  // ------- VISUALIZER --------- //
  
  //! Add all hit points to the visualizer plug-in, and color them by their r-g-b color
  /**
   * \param[in] visualizer Pointer to the Visualizer plugin object.
    * \param[in] pointsize Size of scan point in font points.
  */
  void addHitsToVisualizer( Visualizer* visualizer, const uint pointsize ) const;
  
  //! Add all hit points to the visualizer plug-in, and color them by a hit scalar data value
  /**
   * \param[in] visualizer Pointer to the Visualizer plugin object.
    * \param[in] pointsize Size of scan point in font points.
    * \param[in] color_value Label for scalar hit data
  */
  void addHitsToVisualizer( Visualizer* visualizer, const uint pointsize, const char* color_value ) const;

  //! Add all grid cells to the visualizer plug-in
  /**
   * \param[in] visualizer Pointer to the Visualizer plugin object.
   */
  void addGridToVisualizer( Visualizer* visualizer ) const;

  //! Form an axis-aligned bounding box for all hit points in the point cloud
  /**
   * \param[out] boxmin Coordinates of the bounding box vertex in the (-x,-y,-z) direction
   * \param[out] boxmax Coordinates of the bounding box vertex in the (+x,+y,+z) direction
  */
  void getHitBoundingBox( helios::vec3& boxmin, helios::vec3& boxmax ) const;

  //! Filter scan by imposing a maximum distance from the scanner
  /**
   * \param[in] maxdistance Maximum hit point distance from scanner
   */
  void distanceFilter( const float maxdistance );
  
  //! Filter scan by imposing a minimum reflectance value
  /**
   * \param[in] minreflectance Miniimum hit point reflectance value
   * \note If `reflectance' data was not provided for a hit point when calling \ref Scan::addHitPoint(), the point will not be filtered.
  */
  void reflectanceFilter( const float minreflectance );

  //! Filter hit points based on a scalar field given by a column in the ASCII data
  /**
   * \param[in] scalar_field Name of a scalar field defined in the ASCII point cloud data (e.g., "reflectance")
   * \param[in] threshold Value for filter threshold
   * \param[in] comparator Points will be filtered if "scalar (comparator) threshold", where (comparator) is one of ">", "<", or "="
   * \note As an example, imagine we wanted to remove all hit points where the reflectance is less than -10. In this case we would call scalarFilter( "reflectance", -10, "<" );
  */
  void scalarFilter( const char* scalar_field, const float threshold, const char* comparator );

  // -------- GRID ----------- //

  //! Use a global index of a grid cell (ranging from 0 to Ncells-1) to retrieve the local index in the x-, y-, and z-directions
  /**
   * \param[in] index Global grid cell index - ranges from 0 to Ncells-1
   */
  helios::int3 gridindex2ijk( const int index ) const;

  //! Use a grid cell's local index in the x-, y-, and z-directions to retrieve the cells' global index (ranging from 0 to Ncells-1)
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   */
  int gridijk2index( const helios::int3 ijk ) const;

  //! Get the (x,y,z) coordinate of the grid center
  helios::vec3 getGridCenter( void ) const;

  //! Get the size of the global grid in the x-, y-, and z-directions
  helios::vec3 getGridExtent( void ) const;
  
  //! Get the total number of cells in the grid
  helios::int3 getGridResolution( void ) const;

  //! Get the azimuthal rotation angle of the grid about its center point
  float getGridRotation() const;

  //! Get the center of the (i,j,k)th grid cell
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   */
  helios::vec3 getCellCenter( const helios::int3 ijk ) const;

  //! Get the size of grid cells 
  helios::vec3 getCellSize() const;

  //! Determine the grid cell in which each hit point resides for the whole point cloud */
  /**
    *\note This method does not return a value, rather, it set the Scan variable `hit_vol' which is queried by the function `Scan::getHitGridCell()'.
   */
  void calculateHitGridCell();

  // ------- SYNTHETIC SCAN ------ //

  //! Run a discrete return synthetic LiDAR scan based on scan parameters given in an XML file
  /**
   * \param[in] context Pointer to the Helios context
   * \param[in] xml_file Path to an XML file with LiDAR scan and grid information
  */
  void syntheticScan( helios::Context* context, const char* xml_file );

  //! Run a full-waveform synthetic LiDAR scan based on scan parameters given in an XML file (returns multiple laser hit points per pulse)
  /**
   * \param[in] context Pointer to the Helios context
   * \param[in] xml_file Path to an XML file with LiDAR scan and grid information
   * \param[in] rays_per_pulse Number of ray launches per laser pulse direction
   * \param[in] pulse_distance_threshold Threshold distance for determining laser hit locations. Hits within pulse_distance_threshold of each other will be grouped into a single hit.
   * \note Calling syntheticScan() with rays_per_pulse=1 will effectively run a discrete return synthetic scan.
  */
  void syntheticScan( helios::Context* context, const char* xml_file, const int rays_per_pulse, const float pulse_distance_threshold );

  //! Calculate the surface area of all primitives in the context
  /**
   * \param[in] context Pointer to the Helios context
  */
  void calculateSyntheticLeafArea( helios::Context* context );

  // -------- LEAF AREA -------- //

  //! Set the leaf area of a grid cell in m^2
  /**
   *
   * \param[in] area Leaf area in cell in m^2
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   */
  void setCellLeafArea( const float area, const helios::int3 ijk );

  //! Get the leaf area of a grid cell in m^2
  /**
   * \param [in] index Index of a grid cell in the x-, y-, and z-directions.
   */
  float getCellLeafArea( const helios::int3 ijk ) const;

  //! Get the leaf area density of a grid cell in 1/m
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   */
  float getCellLeafAreaDensity( const helios::int3 ijk ) const;

  //! Set the cell transmission probability counts
  /**
   * \param[in] P_denom Number of rays reaching voxel (denominator of P)
   * \param[in] P_trans Number of rays transmitted through voxel (numerator of P)
   * \param[in] ijk Index of a grid cell in the x-, y-, and z-directions.
   */
  void setCellTransmissionProbability( const int P_denom, const int P_trans, const helios::int3 ijk );

  //! Get the cell transmission probability counts
  /**
   * \param[in] index Index of a grid cell in the x-, y-, and z-directions.
   * \param[out] P_denom Number of rays reaching voxel (denominator of P)
   * \param[out] P_trans Number of rays transmitted through voxel (numerator of P)
  */
  void getCellTransmissionProbability( const helios::int3 ijk, int& P_denom, int& P_trans ) const;

  //! Set the average ray propagation distance in meters
  /**
   * \param[in] r_bar Average ray propagation distance
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   */
  void setCellRbar( const float r_bar, const helios::int3 ijk );

  //! Get the average ray propagation distance in meters
  /**
   * \param [in] index Index of a grid cell in the x-, y-, and z-directions.
   */
  float getCellRbar( const helios::int3 ijk ) const;

  //! Calculate the leaf area for each grid volume
  /**
   * \param[in] Gtheta G-function value to be assumed constant across all cells.
   */
  void calculateLeafAreaGPU( const float Gtheta );

  //! Calculate the leaf area for each grid volume
  /**
   * \param[in] Gtheta G-function value to be assumed constant across all cells.
   * \param [in] minVoxelHits Minimum number of allowable LiDAR hits per voxel. If the total number of hits in a voxel is less than minVoxelHits, the calculated leaf area will be set to zero.
   */
  void calculateLeafAreaGPU( const float Gtheta, const int minVoxelHits );

  // -------- HEIGHT MODEL -------- //
  
  //! Determine the ground and vegetation height for each x-y grid cell. Inputs to this function are parameters for applying the RANSAC algorithm.
  /**
   * \param[in] maxIter Maximum number of iterations to find best fit plane.
   * \param[in] threshDist_ground Maximum distance from fitted plane to be considered an inlier - for ground surface model.
   * \param[in] inlierRatio_ground Minimum fraction of total points that must be inliers to consider the fitted plane valid - for ground surface model.
   * \param[in] threshDist_vegetation Maximum distance from fitted plane to be considered an inlier - for vegetation height model.
   * \param[in] inlierRatio_vegetation Minimum fraction of total points that must be inliers to consider the fitted plane valid - for vegetation height model.
  */
  void generateHeightModel( const int maxIter, const float threshDist_ground, const float inlierRatio_ground, const float threshDist_vegetation, const float inlierRatio_vegetation );

  //! Set the height of the vegetation at the (x,y) location of this gridcell. 
  /**
   * \param[in] height Average height of vegetation measured from the ground in meters.
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
  */
  void setCellVegetationHeight( const float height, const helios::int2 ij );

  //! Get the height of the vegetation at the (x,y) location of this gridcell. 
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   * \return Average height of vegetation measured from the ground in meters.
  */
  float getCellVegetationHeight( const helios::int2 ij )const;

  //! Set the height of the highest hit point at the (x,y) location of this gridcell. 
  /**
   * \param[in] height Maximum height of hit points at the (x,y) location of this gridcell.
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
  */
  void setCellMaximumHitHeight( const float height, const helios::int2 ij );

  //! Get the height of the highest hit point at the (x,y) location of this gridcell.
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   * \return Average height of vegetation measured from the ground in meters.
  */
  float getCellMaximumHitHeight( const helios::int2 ij ) const;

  //! Set the height of the ground at the (x,y) location of this gridcell. 
  /**
   * \param[in] height Height of the ground in meters (in the coordinate system of the point cloud).
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
  */
  void setCellGroundHeight( const float height, const helios::int2 ij );

  //! Get the height of the ground at the (x,y) location of this gridcell. 
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   * \return Height of the ground in meters (in the coordinate system of the point cloud).
  */
  float getCellGroundHeight( const helios::int2 ij ) const;

  //Run calculations for ground cover fraction
  void calculateCoverFraction( void );

  //! Set the ground cover fraction at the (x,y) location of this gridcell. 
  /**
   * \param[in] cover_fraction Ground cover fraction.
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
  */
  void setCellCoverFraction( const float cover_fraction, const helios::int2 ij );

  //! Get the height of the ground at the (x,y) location of this gridcell. 
  /**
   * \param [in] ijk Index of a grid cell in the x-, y-, and z-directions.
   * \return Ground cover fraction.
  */
  float getCellCoverFraction( const helios::int2 ij ) const;
  
};


#endif
