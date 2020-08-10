/** \file "CanopyGenerator.h" Primary header file for canopy geometry generator plug-in.
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

#ifndef __CANOPYGENERATOR__
#define __CANOPYGENERATOR__

#include "Context.h"
#include <random>

//! Parameters defining the homogeneous canopy
struct HomogeneousCanopyParameters{

  //! Default constructor
  HomogeneousCanopyParameters( void );

  //! Length of leaf in x- and y- directions (prior to rotation)
  helios::vec2 leaf_size;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves. If left empty, no texture will be used.
  std::string leaf_texture_file;

  //! Leaf color if no texture map file is provided.
  helios::RGBcolor leaf_color;

  //! One-sided leaf area index of the canopy.
  float leaf_area_index;

  //! Height of the canopy
  float canopy_height;

  //! Horizontal extent of the canopy in the x- and y-directions.
  helios::vec2 canopy_extent;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the canopy with spherical crowns
struct SphericalCrownsCanopyParameters{

  //! Default constructor
  SphericalCrownsCanopyParameters( void );

  //! Length of leaf in x- and y- directions (prior to rotation)
  helios::vec2 leaf_size;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves. If left empty, no texture will be used.
  std::string leaf_texture_file;

  //! Leaf color if no texture map file is provided.
  helios::RGBcolor leaf_color;

  //! One-sided leaf area density within spherical crowns.
  float leaf_area_density;

  //! Radius of the spherical crowns
  float crown_radius;

  //! Specifies whether to use a uniformly spaced canopy (canopy_configuration="uniform") or a randomly arranged canopy with non-overlapping crowns (canopy_configuration="random").
  std::string canopy_configuration;
  
  //! Spacing between adjacent crowns in the x- and y-directions. Note that if canopy_configuration='random' this is the average spacing.
  helios::vec2 plant_spacing;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the grapevine canopy with vertical shoot positioned (VSP) trellis
struct VSPGrapevineParameters{

  //! Default constructor
  VSPGrapevineParameters( void );

  //! Maximum width of leaves. Leaf width increases logarithmically from the shoot tip, so leaf_width is the width at the base of the shoot.
  float leaf_width;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Path to texture map file for trunks/branches.
  std::string wood_texture_file;

  //! Number of radial subdivisions for trunk/cordon/shoot tubes
  int wood_subdivisions;
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Distance between the ground and top of trunks
  float trunk_height;

  //! Radius of the trunk at the widest point
  float trunk_radius;

  //! Distance between the ground and cordon. Note - must be greater than or equal to the trunk height.
  float cordon_height;

  //! Radius of cordon branches.
  float cordon_radius;
  
  //! Length of shoots.
  float shoot_length;

  //! Radius of shoot branches.
  float shoot_radius;

  //! Number of shoots on each cordon.
  uint shoots_per_cordon;

  //! Spacing between adjacent leaves as a fraction of the local leaf width. E.g., leaf_spacing_fraction = 1 would give a leaf spacing equal to the leaf width.
  float leaf_spacing_fraction;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Radius of grape berries
  float grape_radius;

  //! Maximum horizontal radius of grape clusters
  float cluster_radius;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the grapevine canopy with a split (quad) trellis
struct SplitGrapevineParameters{

  //! Default constructor
  SplitGrapevineParameters( void );

  //! Maximum width of leaves. Leaf width increases logarithmically from the shoot tip, so leaf_width is the width at the base of the shoot.
  float leaf_width;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Path to texture map file for trunks/branches.
  std::string wood_texture_file;

  //! Number of radial subdivisions for trunk/cordon/shoot tubes
  int wood_subdivisions;
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Distance between the ground and top of trunks
  float trunk_height;

  //! Radius of the trunk at the widest point
  float trunk_radius;

  //! Distance between the ground and cordon. Note - must be greater than or equal to the trunk height.
  float cordon_height;

  //! Radius of cordon branches.
  float cordon_radius;

  //! Spacing between two opposite cordons
  float cordon_spacing;
  
  //! Length of shoots.
  float shoot_length;

  //! Radius of shoot branches.
  float shoot_radius;

  //! Number of shoots on each cordon.
  uint shoots_per_cordon;

  //! Average angle of the shoot at the base (shoot_angle_base=0 points shoots upward; shoot_angle_base=M_PI points shoots downward and makes a Geneva Double Curtain)
  float shoot_angle_base;
  
  //! Average angle of the shoot at the tip (shoot_angle=0 is a completely vertical shoot; shoot_angle=M_PI is a downward-pointing shoot)
  float shoot_angle_tip;

  //! Spacing between adjacent leaves as a fraction of the local leaf width. E.g., leaf_spacing_fraction = 1 would give a leaf spacing equal to the leaf width.
  float leaf_spacing_fraction;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Radius of grape berries
  float grape_radius;

  //! Maximum horizontal radius of grape clusters
  float cluster_radius;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the grapevine canopy with unilateral trellis
struct UnilateralGrapevineParameters{

  //! Default constructor
  UnilateralGrapevineParameters( void );

  //! Maximum width of leaves. Leaf width increases logarithmically from the shoot tip, so leaf_width is the width at the base of the shoot.
  float leaf_width;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Path to texture map file for trunks/branches.
  std::string wood_texture_file;

  //! Number of radial subdivisions for trunk/cordon/shoot tubes
  int wood_subdivisions;
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Distance between the ground and top of trunks
  float trunk_height;

  //! Radius of the trunk at the widest point
  float trunk_radius;

  //! Distance between the ground and cordon. Note - must be greater than or equal to the trunk height.
  float cordon_height;

  //! Radius of cordon branches.
  float cordon_radius;
  
  //! Length of shoots.
  float shoot_length;

  //! Radius of shoot branches.
  float shoot_radius;

  //! Number of shoots on each cordon.
  uint shoots_per_cordon;

  //! Spacing between adjacent leaves as a fraction of the local leaf width. E.g., leaf_spacing_fraction = 1 would give a leaf spacing equal to the leaf width.
  float leaf_spacing_fraction;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Radius of grape berries
  float grape_radius;

  //! Maximum horizontal radius of grape clusters
  float cluster_radius;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};


//! Parameters defining the grapevine canopy with goblet (vent a taille) trellis
struct GobletGrapevineParameters{

  //! Default constructor
  GobletGrapevineParameters( void );

  //! Maximum width of leaves. Leaf width increases logarithmically from the shoot tip, so leaf_width is the width at the base of the shoot.
  float leaf_width;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Path to texture map file for trunks/branches.
  std::string wood_texture_file;

  //! Number of radial subdivisions for trunk/cordon/shoot tubes
  int wood_subdivisions;
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Distance between the ground and top of trunks
  float trunk_height;

  //! Radius of the trunk at the widest point
  float trunk_radius;

  //! Distance between the ground and cordon. Note - must be greater than or equal to the trunk height.
  float cordon_height;

  //! Radius of cordon branches.
  float cordon_radius;
  
  //! Length of shoots.
  float shoot_length;

  //! Radius of shoot branches.
  float shoot_radius;

  //! Number of shoots on each cordon.
  uint shoots_per_cordon;

  //! Spacing between adjacent leaves as a fraction of the local leaf width. E.g., leaf_spacing_fraction = 1 would give a leaf spacing equal to the leaf width.
  float leaf_spacing_fraction;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Radius of grape berries
  float grape_radius;

  //! Maximum horizontal radius of grape clusters
  float cluster_radius;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the white spruce
struct WhiteSpruceCanopyParameters{

  //! Default constructor
  WhiteSpruceCanopyParameters( void );

  //! Width of needles
  float needle_width;

  //! Length of needles
  float needle_length;

  //! Number of sub-division segments per needle
  helios::int2 needle_subdivisions;

  //! Color of needles
  helios::RGBcolor needle_color;

  //! Path to texture map file for trunks/branches.
  std::string wood_texture_file;

  //! Number of radial subdivisions for trunk/cordon/shoot tubes
  int wood_subdivisions;

  //! Distance between the ground and top of trunks
  float trunk_height;

  //! Radius of the trunk at the base
  float trunk_radius;

  //! Height at which branches start
  float base_height;

  //! Radius of the crown at the base
  float crown_radius;

  //! Radius of shoot branches.
  float shoot_radius;

  //! Vertical spacing between branching levels
  float level_spacing;

  //! Number of primary branches on the bottom level
  int branches_per_level;

  //! Maximum shoot angle
  float shoot_angle;

  //! Specifies whether to use a uniformly spaced canopy (canopy_configuration="uniform") or a randomly arranged canopy with non-overlapping crowns (canopy_configuration="random").
  std::string canopy_configuration;
  
  //! Spacing between adjacent crowns in the x- and y-directions. Note that if canopy_configuration='random' this is the average spacing.
  helios::vec2 plant_spacing;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;
  
};


class CanopyGenerator{
 public:

  //! Canopy geometry generator constructor
  /**  \param[in] "context" Pointer to the Helios context
  */
  CanopyGenerator( helios::Context* context );

  //! Unit testing routine
  int selfTest( void );

  //! Build a canopy consisting of a homogeneous volume of leaves
  /** \param[in] "params" Structure containing parameters for homogeneous canopy.
   */
  void buildCanopy( const HomogeneousCanopyParameters params );

  //! Build a canopy consisting of spherical crowns filled with homogeneous leaves.
  /** \param[in] "params" Structure containing parameters for spherical crown canopy.
   */
  void buildCanopy( const SphericalCrownsCanopyParameters params );

  //! Build a canopy consisting of grapevines on VSP trellis.
  /** \param[in] "params" Structure containing parameters for VSP grapevine canopy.
   */
  void buildCanopy( const VSPGrapevineParameters params );

  //! Build a canopy consisting of grapevines on split trellis.
  /** \param[in] "params" Structure containing parameters for split trellis grapevine canopy.
   */
  void buildCanopy( const SplitGrapevineParameters params );

  //! Build a canopy consisting of grapevines on unilateral trellis.
  /** \param[in] "params" Structure containing parameters for unilateral grapevine canopy.
   */
  void buildCanopy( const UnilateralGrapevineParameters params );

  //! Build a canopy consisting of grapevines on Goblet trellis.
  /** \param[in] "params" Structure containing parameters for Goblet grapevine canopy.
   */
  void buildCanopy( const GobletGrapevineParameters params );

  //! Build a canopy consisting of white spruce trees
  /** \param[in] "params" Structure containing parameters for white spruce canopy.
   */
  void buildCanopy( const WhiteSpruceCanopyParameters params );

  //! Build a ground consisting of texture sub-tiles and sub-patches, which can be different sizes
  /** \param[in] "ground_origin" x-, y-, and z-position of the ground center point.
      \param[in] "ground_extent" Width of the ground in the x- and y-directions.
      \param[in] "texture_subtiles" Number of sub-divisions of the ground into texture map tiles in the x- and y-directions.
      \param[in] "texture_subpatches" Number of sub-divisions of each texture tile into sub-patches in the x- and y-directions.
      \param[in] "ground_texture_file" Path to file used for tile texture mapping.
  */
  void buildGround( const helios::vec3 ground_origin, const helios::vec2 ground_extent, const helios::int2 texture_subtiles, const helios::int2 texture_subpatches, const char* ground_texture_file  );

  //! Build a ground with azimuthal rotation consisting of texture sub-tiles and sub-patches, which can be different sizes
  /** \param[in] "ground_origin" x-, y-, and z-position of the ground center point.
      \param[in] "ground_extent" Width of the ground in the x- and y-directions.
      \param[in] "texture_subtiles" Number of sub-divisions of the ground into texture map tiles in the x- and y-directions.
      \param[in] "texture_subpatches" Number of sub-divisions of each texture tile into sub-patches in the x- and y-directions.
      \param[in] "ground_texture_file" Path to file used for tile texture mapping.
      \param[in] "ground_rotation" Azimuthal rotation angle of ground in radians.
  */
  void buildGround( const helios::vec3 ground_origin, const helios::vec2 ground_extent, const helios::int2 texture_subtiles, const helios::int2 texture_subpatches, const char* ground_texture_file, const float ground_rotation  );
 
  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree trunk
  /** \param[in] "TreeID" Identifer of tree.
  */
  std::vector<uint> getTrunkUUIDs( const uint TreeID );

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree branches
  /** \param[in] "TreeID" Identifer of tree.
  */
  std::vector<uint> getBranchUUIDs( const uint TreeID );

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree leaves
  /** \param[in] "TreeID" Identifer of tree.
      \note The first index is the leaf, second index is the UUIDs making up the sub-primitives of the leaf (if appplicable).
  */
  std::vector<std::vector<uint> > getLeafUUIDs( const uint TreeID );

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree fruit
  /** \param[in] "TreeID" Identifer of tree.
      \note First index is the cluster of fruit (if applicable), second index is the fruit, third index is the UUIDs making up the sub-primitives of the fruit.
  */
  std::vector<std::vector<std::vector<uint> > > getFruitUUIDs( const uint TreeID );

  //! Get the unique universal identifiers (UUIDs) for all primitives that make up the tree
  /** \param[in] "TreeID" Identifer of tree.
  */
  std::vector<uint> getAllUUIDs( const uint TreeID );

  //! Get the current number of plants added to the Canopy Generator
  uint getPlantCount( void );

  //! Seed the random number generator. This can be useful for generating repeatable trees, say, within a loop.
  /** \param[in] "seed" Random number seed */
  void seedRandomGenerator( const uint seed );

  //! Disable standard messages from being printed to screen (default is to enable messages)
  void disableMessages( void );

  //! Enable standard messages to be printed to screen (default is to enable messages)
  void enableMessages( void );

  //---------- PLANT GEOMETRIES ------------ //

  //! Function to add an individual grape berry cluster
  /* \param[in] "position" Cartesian (x,y,z) position of the cluster main stem.
     \param[in] "grape_rad" Maximum grape berry radius.
     \param[in] "cluster_rad" Radius of berry cluster at widest point.
     \return 2D vector of primitive UUIDs. The first index is the UUID for each primitive (triangles) comprising individual berries, second index corresponds to each berry in the cluster.
  */
  std::vector<std::vector<uint> > addGrapeCluster( helios::vec3 position, float grape_rad, float cluster_rad );

//! Function to add an individual grapevine plant on a vertical shoot positioned (VSP) trellis.
/* \param[in] "params" Set of parameters defining grapevine plant.
   \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
*/ 
void grapevineVSP( const VSPGrapevineParameters params, const helios::vec3 origin );

//! Function to add an individual grapevine plant on a split trellis.
/* \param[in] "params" Set of parameters defining grapevine plant.
   \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
*/ 
void grapevineSplit( const SplitGrapevineParameters params, const helios::vec3 origin );

//! Function to add an individual grapevine plant on a unilateral trellis.
/* \param[in] "params" Set of parameters defining grapevine plant.
   \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
*/ 
void grapevineUnilateral( const UnilateralGrapevineParameters params, const helios::vec3 origin );

//! Function to add an individual grapevine plant on a goblet (vent a taille) trellis.
/* \param[in] "params" Set of parameters defining grapevine plant.
   \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
*/ 
void grapevineGoblet( const GobletGrapevineParameters params, const helios::vec3 origin );

  //! Function to add an individual white spruce tree
  /* \param[in] "params" Set of parameters defining white spruce tree.
     \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
  */ 
  void whitespruce( const WhiteSpruceCanopyParameters params, const helios::vec3 origin );

 private:

  helios::Context* context;

  //! UUIDs for trunk primitives
  /* \note First index in the vector is the plant, second index is the UUIDs making up the trunk for that plant. */
  std::vector<std::vector<uint> > UUID_trunk;

  //! UUIDs for branch primitives
  /* \note First index in the vector is the plant, second index is the UUIDs making up the branches for that plant. */
  std::vector<std::vector<uint> > UUID_branch;

  //! UUIDs for leaf primitives
  /* \note First index in the vector is the plant, second index is the leaf, third index is the UUIDs making up the sub-primitives of the leaf (if appplicable). */
  std::vector<std::vector<std::vector<uint> > > UUID_leaf;

  //! UUIDs for fruit primitives
  /* \note First index in the vector is the plant, second index is the cluster of fruit (if applicable), third index is the fruit, fourth index is the UUIDs making up the sub-primitives making of the fruit. */
  std::vector<std::vector<std::vector<std::vector<uint> > > > UUID_fruit;

  //! UUIDs for ground primitives
  std::vector<uint> UUID_ground; 

  float sampleLeafAngle( const std::vector<float> leafAngleDist );
 
  std::minstd_rand0 generator;

  bool printmessages;

};

float getVariation( float V, std::minstd_rand0& generator );

helios::vec3 interpolateTube( const std::vector<helios::vec3> P, const float frac );

float interpolateTube( const std::vector<float> P, const float frac );

#endif
