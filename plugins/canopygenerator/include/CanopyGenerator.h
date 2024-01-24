/** \file "CanopyGenerator.h" Primary header file for canopy geometry generator plug-in.
    
    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef CANOPY_GENERATOR
#define CANOPY_GENERATOR

#include "Context.h"
#include <random>

//! Parameters defining the homogeneous canopy
struct HomogeneousCanopyParameters{

  //! Default constructor
  HomogeneousCanopyParameters();

  //! Length of leaf in x- and y- directions (prior to rotation)
  helios::vec2 leaf_size;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves. If left empty, no texture will be used.
  std::string leaf_texture_file;

  //! Leaf color if no texture map file is provided.
  helios::RGBcolor leaf_color;

  //! Leaf angle distribution - one of "spherical", "uniform", "erectophile", "planophile", "plagiophile", "extremophile"
  std::string leaf_angle_distribution;

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

  //! String specifying whether leaves should be placed so that leaf edges do not fall outside the specified canopy dimensions ("z", "xyz", or "none")
  std::string buffer;
  
};

//! Parameters defining the canopy with spherical crowns
struct SphericalCrownsCanopyParameters{

  //! Default constructor
  SphericalCrownsCanopyParameters();

  //! Length of leaf in x- and y- directions (prior to rotation)
  helios::vec2 leaf_size;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves. If left empty, no texture will be used.
  std::string leaf_texture_file;

  //! Leaf color if no texture map file is provided.
  helios::RGBcolor leaf_color;

  //! Leaf angle distribution - one of "spherical", "uniform", "erectophile", "planophile", "plagiophile", "extremophile"
  std::string leaf_angle_distribution;

  //! One-sided leaf area density within spherical crowns.
  float leaf_area_density;

  //! Radius of the spherical crowns
  helios::vec3 crown_radius;

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

//! Parameters defining the canopy with conical crowns
struct ConicalCrownsCanopyParameters{

    //! Default constructor
    ConicalCrownsCanopyParameters();

    //! Length of leaf in x- and y- directions (prior to rotation)
    helios::vec2 leaf_size;

    //! Number of sub-division segments per leaf
    helios::int2 leaf_subdivisions;

    //! Path to texture map file for leaves. If left empty, no texture will be used.
    std::string leaf_texture_file;

    //! Leaf color if no texture map file is provided.
    helios::RGBcolor leaf_color;

    //! Leaf angle distribution - one of "spherical", "uniform", "erectophile", "planophile", "plagiophile", "extremophile"
    std::string leaf_angle_distribution;

    //! One-sided leaf area density within spherical crowns.
    float leaf_area_density;

    //! Radius of the conical crowns at the base
    float crown_radius;

    //! Height of the conical crowns
    float crown_height;

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
  VSPGrapevineParameters();

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

  //! Maximum height of grape clusters along the shoot as a fraction of the total shoot length
  float cluster_height_max;

  //! Color of grapes
  helios::RGBcolor grape_color;

  //! Number of azimuthal and zenithal subdivisions making up berries (will result in roughly 2*(grape_subdivisions)^2 triangles per grape berry)
  uint grape_subdivisions;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the grapevine canopy with a split (quad) trellis
struct SplitGrapevineParameters{

  //! Default constructor
  SplitGrapevineParameters();

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

  //! Maximum height of grape clusters along the shoot as a fraction of the total shoot length
  float cluster_height_max;

  //! Color of grapes
  helios::RGBcolor grape_color;

  //! Number of azimuthal and zenithal subdivisions making up berries (will result in roughly grape_subdivisions^2 triangles per grape berry)
  uint grape_subdivisions;
  
  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the grapevine canopy with unilateral trellis
struct UnilateralGrapevineParameters{

  //! Default constructor
  UnilateralGrapevineParameters();

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

  //! Maximum height of grape clusters along the shoot as a fraction of the total shoot length
  float cluster_height_max;

  //! Color of grapes
  helios::RGBcolor grape_color;

  //! Number of azimuthal and zenithal subdivisions making up berries (will result in roughly grape_subdivisions^2 triangles per grape berry)
  uint grape_subdivisions;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};


//! Parameters defining the grapevine canopy with goblet (vent a taille) trellis
struct GobletGrapevineParameters{

  //! Default constructor
  GobletGrapevineParameters();

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

  //! Maximum height of grape clusters along the shoot as a fraction of the total shoot length
  float cluster_height_max;

  //! Color of grapes
  helios::RGBcolor grape_color;

  //! Number of azimuthal and zenithal subdivisions making up berries (will result in roughly grape_subdivisions^2 triangles per grape berry)
  uint grape_subdivisions;

  //! 
  std::vector<float> leaf_angle_PDF;
  
};

//! Parameters defining the white spruce
struct WhiteSpruceCanopyParameters{

  //! Default constructor
  WhiteSpruceCanopyParameters();

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

//! Parameters defining the tomato plant canopy
struct TomatoParameters{

  //! Default constructor
  TomatoParameters();

  //! Maximum width of leaves. 
  float leaf_length;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Color of shoots
  helios::RGBcolor shoot_color;

  //! Number of radial subdivisions for shoot tubes
  int shoot_subdivisions;
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Height of the plant
  float plant_height;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Radius of tomato fruit
  float fruit_radius;

  //! Color of tomato fruit
  helios::RGBcolor fruit_color;

  //! Number of azimuthal and zenithal subdivisions making up fruit (will result in roughly fruit_subdivisions^2 triangles per fruit)
  uint fruit_subdivisions;
  
};

//! Parameters defining the strawberry plant canopy
struct StrawberryParameters{

  //! Default constructor
  StrawberryParameters();

  //! Maximum width of leaves. 
  float leaf_length;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Color of stems
  helios::RGBcolor stem_color;

  //! Number of radial subdivisions for stem tubes
  int stem_subdivisions;

  //! Number of stems per plant
  int stems_per_plant;

  //! Radius of stems
  float stem_radius;
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Height of the plant
  float plant_height;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Radius of strawberry fruit
  float fruit_radius;

  //! Texture map for strawberry fruit
  std::string fruit_texture_file;

  //! Number of azimuthal and zenithal subdivisions making up fruit (will result in roughly fruit_subdivisions^2 triangles per fruit)
  uint fruit_subdivisions;

  //! Number of strawberry clusters per plant stem. Clusters randomly have 1, 2, or 3 berries.
  float clusters_per_stem;
  
};

//! Parameters defining the walnut tree canopy
struct WalnutCanopyParameters{

  //! Default constructor
  WalnutCanopyParameters();

  //! Maximum length of leaves along midrib. 
  float leaf_length;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Path to texture map file for wood/branches.
  std::string wood_texture_file;

  //! Number of radial subdivisions for branch tubes
  int wood_subdivisions;

  //! Radius of trunk
  float trunk_radius;

  //! Height of the trunk
  float trunk_height;

  //! Average length of branches in each recursive branch level. For example, the first (.x) value is the length of branches emanating from the trunk, the second (.y) is the the length of branches emanating from the first branching level.
  helios::vec3 branch_length;
  
  
  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Radius of walnuts
  float fruit_radius;

  //! Texture map for walnut fruit
  std::string fruit_texture_file;

  //! Number of azimuthal and zenithal subdivisions making up fruit (will result in roughly fruit_subdivisions^2 triangles per fruit)
  uint fruit_subdivisions;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;
  
};

//! Parameters defining Sorghum plant canopy
struct SorghumCanopyParameters{

//! Default constructor
    SorghumCanopyParameters();

//! Sorghum categorized into 5 stages; 1 - Three leaf stage, 2 - Five leaf stage, 3 - Panicle initiation and flag leaf emergency, 4 - Booting, and flowering, 5 - Maturity;  Input of a value other than 1-5 will output stage 5
    int sorghum_stage;

    // STAGE 1

//! Length of the sorghum stem for stage 1
    float s1_stem_length;

//! Radius of the sorghum stem for stage 1
    float s1_stem_radius;

//! Number of stem radial subdivisions for stage 1
    float s1_stem_subdivisions;

//! Length of leaf1 in x- (length) and y- (width) directions (prior to rotation) for stage 1
    helios::vec2 s1_leaf_size1;

//! Length of leaf2 in x- (length) and y- (width) directions (prior to rotation) for stage 1
    helios::vec2 s1_leaf_size2;

//! Length of leaf3 in x- (length) and y- (width) directions (prior to rotation) for stage 1
    helios::vec2 s1_leaf_size3;

//! Leaf1 vertical angle of rotation for stage 1
    float s1_leaf1_angle;

//! Leaf2 vertical angle of rotation for stage 1
    float s1_leaf2_angle;

//! Leaf3 vertical angle of rotation for stage 1
    float s1_leaf3_angle;

//! Number of sub-division segments of leaf in the x- (length) and y- (width) direction for stage 1
    helios::int2 s1_leaf_subdivisions;

//! Texture map for sorghum leaf1 for stage 1
    std::string s1_leaf_texture_file;

    // STAGE 2

//! Length of the sorghum stem for stage 2
    float s2_stem_length;

//! Radius of the sorghum stem for stage 2
    float s2_stem_radius;

//! Number of stem radial subdivisions for stage 2
    float s2_stem_subdivisions;

//! Length of leaf1 in x- (length) and y- (width) directions (prior to rotation) for stage 2
    helios::vec2 s2_leaf_size1;

//! Length of leaf2 in x- (length) and y- (width) directions (prior to rotation) for stage 2
    helios::vec2 s2_leaf_size2;

//! Length of leaf3 in x- (length) and y- (width) directions (prior to rotation) for stage 2
    helios::vec2 s2_leaf_size3;

//! Length of leaf4 in x- (length) and y- (width) directions (prior to rotation) for stage 2
    helios::vec2 s2_leaf_size4;

//! Length of leaf5 in x- (length) and y- (width) directions (prior to rotation) for stage 2
    helios::vec2 s2_leaf_size5;

//! Leaf1 vertical angle of rotation for stage 2
    float s2_leaf1_angle;

//! Leaf2 vertical angle of rotation for stage 2
    float s2_leaf2_angle;

//! Leaf3 vertical angle of rotation for stage 2
    float s2_leaf3_angle;

//! Leaf4 vertical angle of rotation for stage 2
    float s2_leaf4_angle;

//! Leaf5 vertical angle of rotation for stage 2
    float s2_leaf5_angle;

//! Number of sub-division segments of leaf in the x- (length) and y- (width) direction for stage 2
    helios::int2 s2_leaf_subdivisions;

//! Texture map for sorghum leaf for stage 2
    std::string s2_leaf_texture_file;

    // STAGE 3

//! Length of the sorghum stem for stage 3
    float s3_stem_length;

//! Radius of the sorghum stem for stage 3
    float s3_stem_radius;

//! Number of stem radial subdivisions for stage 3
    float s3_stem_subdivisions;

//! Length of leaf in x- (length) and y- (width) directions (prior to rotation) for stage 3
    helios::vec2 s3_leaf_size;

//! Number of sub-division segments of leaf in the x- (length) and y- (width) direction for stage 3
    helios::int2 s3_leaf_subdivisions;

//! Number of leaves along the stem for stage 3
    int s3_number_of_leaves;

//! Mean vertical angle of rotation of leaf for stage 3 in degrees; Standard deviation for the leaves is 5 degrees
    float s3_mean_leaf_angle;

//! Texture map for sorghum leaf for stage 3
    std::string s3_leaf_texture_file;

    // STAGE 4

//! Length of the sorghum stem for stage 4
    float s4_stem_length;

//! Radius of the sorghum stem for stage 4
    float s4_stem_radius;

//! Number of stem radial subdivisions for stage 4
    float s4_stem_subdivisions;

//! Size of panicle in x- and y- directions for stage 4
    helios::vec2 s4_panicle_size;

//! Number of panicle subdivisions for each grain sphere within a panicle, stage 4
    int s4_panicle_subdivisions;

//! Texture map of the panicle for stage 4
    std::string s4_seed_texture_file;

//! Length of leaf in x- (length) and y- (width) directions (prior to rotation) for stage 4
    helios::vec2 s4_leaf_size;

//! Number of sub-division segments of leaf in the x- (length) and y- (width) direction for stage 4
    helios::int2 s4_leaf_subdivisions;

//! Number of leaves for the sorghum plant, stage 4
    int s4_number_of_leaves;

//! Mean vertical angle of rotation of leaf for stage 4 in degrees; Standard deviation for the angle is 5 degrees
    float s4_mean_leaf_angle;

//! Texture map for sorghum leaf, stage 4
    std::string s4_leaf_texture_file;

    // STAGE 5

//! Length of the sorghum stem for stage 5
    float s5_stem_length;

//! Radius of the sorghum stem for stage 5
    float s5_stem_radius;

//! Bend of the stem from mid-section for stage 5. The distance from the mid-section of the stem to the imaginary perpendicular along the origin. i.e stem bend = 0 outputs a straight stem
    float s5_stem_bend;

//! Number of stem radial subdivisions for stage 5
    int s5_stem_subdivisions;

//! Size of panicle in x- and y- directions for stage 5
    helios::vec2 s5_panicle_size;

//! Number of panicle subdivisions for each grain sphere within a panicle, stage 5
    int s5_panicle_subdivisions;

//! Texture map of the panicle for stage 5
    std::string s5_seed_texture_file;

//! Length of leaf in x- (length) and y- (width) directions (prior to rotation) for stage 5
    helios::vec2 s5_leaf_size;

//! Number of sub-division segments of leaf in the x- (length) and y- (width) direction for stage 5
    helios::int2 s5_leaf_subdivisions;

//! Number of leaves for the sorghum plant, stage 5
    int s5_number_of_leaves;

//! Mean vertical angle of rotation of leaf for stage 5 in degrees; Standard deviation for the angle is 10 degrees
    float s5_mean_leaf_angle;

//! Texture map for sorghum leaf, stage 5
    std::string s5_leaf_texture_file;

    // CANOPY
//! Spacing between adjacent plants along the row direction.
    float plant_spacing;

//! Spacing between plant rows.
    float row_spacing;

//! Number of crowns/plants in the x- and y-directions.
    helios::int2 plant_count;

//! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0.
    helios::vec3 canopy_origin;

};

//! Parameters defining the bean plant canopy
struct BeanParameters{

  //! Default constructor
  BeanParameters();

  //! Maximum width of leaves.
  float leaf_length;

  //! Number of sub-division segments per leaf
  helios::int2 leaf_subdivisions;

  //! Path to texture map file for leaves.
  std::string leaf_texture_file;

  //! Color of shoots
  helios::RGBcolor shoot_color;

  //! Number of radial subdivisions for shoot tubes
  int shoot_subdivisions;

  //! Radius of main stem at base
  float stem_radius;

  //! Length of stems before splitting to leaflets
  float stem_length;

  //! Length of the leaflet from base to tip leaf
  float leaflet_length;

  //! Spacing between adjacent plants along the row direction.
  float plant_spacing;

  //! Spacing between plant rows.
  float row_spacing;

  //! Number of crowns/plants in the x- and y-directions.
  helios::int2 plant_count;

  //! Probability that a plant in the canopy germinated
  float germination_probability;

  //! Cartesian (x,y,z) coordinate of the bottom center point of the canopy (i.e., specifying z=0 places the bottom surface of the canopy at z=0).
  helios::vec3 canopy_origin;

  //! Azimuthal rotation of the canopy about the canopy origin. Note that if canopy_rotation is not equal to zero, the plant_spacing and plant_count parameters are defined in the x- and y-directions before rotation.
  float canopy_rotation;

  //! Length of bean pods
  float pod_length;

  //! Color of bean pods
  helios::RGBcolor pod_color;

  //! Number of lengthwise subdivisions making up pods
  uint pod_subdivisions;

};

class CanopyGenerator{
 public:

  //! Canopy geometry generator constructor
  /**
   * \param[in] "context" Pointer to the Helios context
  */
  explicit CanopyGenerator( helios::Context* context );

  //! Unit testing routine
 int selfTest();

  //! Build canopy geometries based on parameters specified in an XML file
  /**
   * \param[in] "filename" Path to XML file to be read
   */
  void loadXML( const char* filename );

  //! Build a canopy consisting of a homogeneous volume of leaves
  /**
   * \param[in] "params" Structure containing parameters for homogeneous canopy.
   */
  void buildCanopy( const HomogeneousCanopyParameters &params );

  //! Build a canopy consisting of spherical crowns filled with homogeneous leaves.
  /**
   * \param[in] "params" Structure containing parameters for spherical crown canopy.
   */
  void buildCanopy( const SphericalCrownsCanopyParameters &params );

  //! Build a canopy consisting of conical crowns filled with homogeneous leaves.
  /**
    * \param[in] "params" Structure containing parameters for conical crown canopy.
   */
  void buildCanopy( const ConicalCrownsCanopyParameters &params );

  //! Build a canopy consisting of grapevines on VSP trellis.
  /**
   * \param[in] "params" Structure containing parameters for VSP grapevine canopy.
   */
  void buildCanopy( const VSPGrapevineParameters &params );

  //! Build a canopy consisting of grapevines on split trellis.
  /**
   * \param[in] "params" Structure containing parameters for split trellis grapevine canopy.
   */
  void buildCanopy( const SplitGrapevineParameters &params );

  //! Build a canopy consisting of grapevines on unilateral trellis.
  /**
   * \param[in] "params" Structure containing parameters for unilateral grapevine canopy.
   */
  void buildCanopy( const UnilateralGrapevineParameters &params );

  //! Build a canopy consisting of grapevines on Goblet trellis.
  /**
   * \param[in] "params" Structure containing parameters for Goblet grapevine canopy.
   */
  void buildCanopy( const GobletGrapevineParameters &params );

  //! Build a canopy consisting of white spruce trees
  /**
   * \param[in] "params" Structure containing parameters for white spruce canopy.
   */
  void buildCanopy( const WhiteSpruceCanopyParameters &params );

  //! Build a canopy consisting of tomato plants
  /**
   * \param[in] "params" Structure containing parameters for tomato canopy.
   */
  void buildCanopy( const TomatoParameters &params );

  //! Build a canopy consisting of strawberry plants
  /**
   * \param[in] "params" Structure containing parameters for strawberry canopy.
   */
  void buildCanopy( const StrawberryParameters &params );

  //! Build a canopy consisting of walnut trees
  /**
   * \param[in] "params" Structure containing parameters for walnut tree canopy.
   */
  void buildCanopy(const WalnutCanopyParameters &params );

  //! Build a canopy consisting of sorghum plants
  /**
   * \param[in] "params" Structure containing parameters for sorghum plant canopy.
   */
    void buildCanopy( const SorghumCanopyParameters &params );

    //! Build a canopy consisting of common bean plants
    /**
        * \param[in] "params" Structure containing parameters for bean canopy.
     */
    void buildCanopy( const BeanParameters &params );

  //! Build a ground consisting of texture sub-tiles and sub-patches, which can be different sizes
  /**
   * \param[in] "ground_origin" x-, y-, and z-position of the ground center point.
   * \param[in] "ground_extent" Width of the ground in the x- and y-directions.
   * \param[in] "texture_subtiles" Number of sub-divisions of the ground into texture map tiles in the x- and y-directions.
   * \param[in] "texture_subpatches" Number of sub-divisions of each texture tile into sub-patches in the x- and y-directions.
   * \param[in] "ground_texture_file" Path to file used for tile texture mapping.
  */

  void buildGround( const helios::vec3 &ground_origin, const helios::vec2 &ground_extent, const helios::int2 &texture_subtiles, const helios::int2 &texture_subpatches, const char* ground_texture_file  );

  //! Build a ground with azimuthal rotation consisting of texture sub-tiles and sub-patches, which can be different sizes
  /**
   * \param[in] "ground_origin" x-, y-, and z-position of the ground center point.
   * \param[in] "ground_extent" Width of the ground in the x- and y-directions.
   * \param[in] "texture_subtiles" Number of sub-divisions of the ground into texture map tiles in the x- and y-directions.
   * \param[in] "texture_subpatches" Number of sub-divisions of each texture tile into sub-patches in the x- and y-directions.
   * \param[in] "ground_texture_file" Path to file used for tile texture mapping.
   * \param[in] "ground_rotation" Azimuthal rotation angle of ground in radians.
  */
  void buildGround( const helios::vec3 &ground_origin, const helios::vec2 &ground_extent, const helios::int2 &texture_subtiles, const helios::int2 &texture_subpatches, const char* ground_texture_file, float ground_rotation  );
 
  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the plant trunk
  /**
   * \param[in] "PlantID" Identifier of plant.
  */
  std::vector<uint> getTrunkUUIDs( uint PlantID );

  //! Get the unique universal identifiers (UUIDs) for all trunk primitives in a single 1D vector
  std::vector<uint> getTrunkUUIDs();

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the plant branches
  /**
   * \param[in] "PlantID" Identifier of plant.
  */
  std::vector<uint> getBranchUUIDs( uint PlantID );

  //! Get the unique universal identifiers (UUIDs) for all branch primitives in a single 1D vector
  std::vector<uint> getBranchUUIDs();

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the plant leaves
  /**
   * \param[in] "PlantID" Identifier of plant.
   * \note The first index is the leaf, second index is the UUIDs making up the sub-primitives of the leaf (if applicable).
  */
  std::vector<std::vector<uint> > getLeafUUIDs( uint PlantID );

  //! Get the unique universal identifiers (UUIDs) for all leaf primitives in a single 1D vector
  std::vector<uint> getLeafUUIDs();

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the tree fruit
  /**
   * \param[in] "PlantID" Identifier of tree.
   * \note First index is the cluster of fruit (if applicable), second index is the fruit, third index is the UUIDs making up the sub-primitives of the fruit.
  */
  std::vector<std::vector<std::vector<uint> > > getFruitUUIDs( uint PlantID );

  //! Get the unique universal identifiers (UUIDs) for all fruit primitives in a single 1D vector
  std::vector<uint> getFruitUUIDs();

  //! Get the unique universal identifiers (UUIDs) for the primitives that make up the ground
  std::vector<uint> getGroundUUIDs();

  //! Get the unique universal identifiers (UUIDs) for all primitives that make up the tree
  /**
   * \param[in] "PlantID" Identifier of tree.
  */
  std::vector<uint> getAllUUIDs( uint PlantID );

  //! Get the current number of plants added to the Canopy Generator
  uint getPlantCount();

  //! Seed the random number generator. This can be useful for generating repeatable trees, say, within a loop.
  /**
   * \param[in] "seed" Random number seed
   */
  void seedRandomGenerator( uint seed );

  //! Disable standard messages from being printed to screen (default is to enable messages)
  void disableMessages();

  //! Enable standard messages to be printed to screen (default is to enable messages)
  void enableMessages();

  //---------- PLANT GEOMETRIES ------------ //

  //! Function to add an individual grape berry cluster
  /**
   * \param[in] "position" Cartesian (x,y,z) position of the cluster main stem.
   * \param[in] "grape_rad" Maximum grape berry radius.
   * \param[in] "cluster_rad" Radius of berry cluster at widest point.
   * \param[in] "grape_subdiv" Number of azimuthal and zenithal berry sub-triangle subdivisions.
   * \return 2D vector of primitive UUIDs. The first index is the UUID for each primitive (triangles) comprising individual berries, second index corresponds to each berry in the cluster.
  */
  std::vector<std::vector<uint> > addGrapeCluster( helios::vec3 position, float grape_rad, float cluster_rad, helios::RGBcolor grape_color, uint grape_subdiv );

//! Function to add an individual grapevine plant on a vertical shoot positioned (VSP) trellis.
/**
 * \param[in] "params" Set of parameters defining grapevine plant.
 * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
 * \return Plant ID of bean plant.
*/ 
uint grapevineVSP(const VSPGrapevineParameters &params, const helios::vec3 &origin );

//! Function to add an individual grapevine plant on a split trellis.
/**
 * \param[in] "params" Set of parameters defining grapevine plant.
 * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
 * \return Plant ID of bean plant.
*/ 
uint grapevineSplit( const SplitGrapevineParameters &params, const helios::vec3 &origin );

//! Function to add an individual grapevine plant on a unilateral trellis.
/**
 * \param[in] "params" Set of parameters defining grapevine plant.
 * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
 * \return Plant ID of bean plant.
*/ 
uint grapevineUnilateral( const UnilateralGrapevineParameters &params, const helios::vec3 &origin );

//! Function to add an individual grapevine plant on a goblet (vent a taille) trellis.
/**
 * \param[in] "params" Set of parameters defining grapevine plant.
 * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
 * \return Plant ID of bean plant.
*/ 
uint grapevineGoblet( const GobletGrapevineParameters &params, const helios::vec3 &origin );

  //! Function to add an individual white spruce tree
  /**
   * \param[in] "params" Set of parameters defining white spruce tree.
   * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
   * \return Plant ID of bean plant.
  */ 
  uint whitespruce( const WhiteSpruceCanopyParameters &params, const helios::vec3 &origin );

  //! Function to add an individual tomato plant
  /**
   * \param[in] "params" Set of parameters defining tomato plants/canopy.
   * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
   * \return Plant ID of bean plant.
  */ 
  uint tomato( const TomatoParameters &params, const helios::vec3 &origin );

  //! Function to add an individual strawberry plant
  /**
   * \param[in] "params" Set of parameters defining strawberry plants/canopy.
   * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
   * \return Plant ID of bean plant.
  */ 
  uint strawberry( const StrawberryParameters &params, const helios::vec3 &origin );

  //! Function to add an individual walnut tree
  /**
   * \param[in] "params" Set of parameters defining walnut trees/canopy.
   * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
   * \return Plant ID of bean plant.
  */ 
  uint walnut(const WalnutCanopyParameters &params, const helios::vec3 &origin );

 //! Function to add an individual sorghum plant
 /**
 * \param[in] "params" Set of parameters defining sorghum plants/canopy.
 * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
  * \return Plant ID of bean plant.
 */
 uint sorghum( const SorghumCanopyParameters &params, const helios::vec3 &origin);

 //! Function to add an individual bean plant
 /**
    * \param[in] "params" Set of parameters defining bean plants/canopy.
    * \param[in] "origin" Cartesian (x,y,z) position of the center of the canopy.
    * \return Plant ID of bean plant.
   */
 uint bean( const BeanParameters &params, const helios::vec3 &origin );

  //! Create primitive data that explicitly labels all primitives according to the plant element they correspond to
  void createElementLabels();

  //! Toggle off primitive data element type labels
  void disableElementLabels();

 private:

  helios::Context* context;

  //! UUIDs for trunk primitives
  /**
   * \note First index in the vector is the plant, second index is the UUIDs making up the trunk for that plant.
   */
  std::vector<std::vector<uint> > UUID_trunk;

  //! UUIDs for branch primitives
  /**
   * \note First index in the vector is the plant, second index is the UUIDs making up the branches for that plant.
   */
  std::vector<std::vector<uint> > UUID_branch;

  //! UUIDs for leaf primitives
  /**
   * \note First index in the vector is the plant, second index is the leaf, third index is the UUIDs making up the sub-primitives of the leaf (if applicable).
   */
  std::vector<std::vector<std::vector<uint> > > UUID_leaf;

  //! UUIDs for fruit primitives
  /**
   * \note First index in the vector is the plant, second index is the cluster of fruit (if applicable), third index is the fruit, fourth index is the UUIDs making up the sub-primitives making of the fruit.
   */
  std::vector<std::vector<std::vector<std::vector<uint> > > > UUID_fruit;

  //! UUIDs for ground primitives
  std::vector<uint> UUID_ground;

  std::vector<uint> leaf_prototype_global;

  float sampleLeafAngle( const std::vector<float> &leafAngleDist );

  float sampleLeafPDF( const char* distribution );
 
  std::minstd_rand0 generator;

  bool printmessages;

  bool enable_element_labels;

  void cleanDeletedUUIDs( std::vector<uint> &UUIDs );

  void cleanDeletedUUIDs( std::vector<std::vector<uint> > &UUIDs );

  void cleanDeletedUUIDs( std::vector<std::vector<std::vector<uint> > > &UUIDs );

};

float getVariation( float V, std::minstd_rand0& generator );

helios::vec3 interpolateTube( const std::vector<helios::vec3> &P, float frac );

float interpolateTube( const std::vector<float> &P, float frac );


#endif
