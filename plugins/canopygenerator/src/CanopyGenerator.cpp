/** \file "CanopyGenerator.cpp" Primary source file for canopy geometry generator plug-in.
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

#include "CanopyGenerator.h"

using namespace helios;

HomogeneousCanopyParameters::HomogeneousCanopyParameters(void){

  leaf_size = make_vec2(0.1,0.1);

  leaf_subdivisions = make_int2(1,1);

  leaf_color = RGB::green;

  leaf_angle_distribution = "spherical";

  leaf_area_index = 1.f;

  canopy_height = 1.f;

  canopy_extent = make_vec2(5,5);

  canopy_origin = make_vec3(0,0,0);
  
}

SphericalCrownsCanopyParameters::SphericalCrownsCanopyParameters(void){

  leaf_size = make_vec2(0.025,0.025);

  leaf_subdivisions = make_int2(1,1);

  leaf_color = RGB::green;

  leaf_angle_distribution = "spherical";

  leaf_area_density = 1.f;

  crown_radius = 0.5f;

  canopy_configuration = "uniform";

  plant_spacing = make_vec2(2.f,2.f);

  plant_count = make_int2(5,5);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0.f;
  
}

VSPGrapevineParameters::VSPGrapevineParameters(void){

  leaf_width = 0.18;

  leaf_subdivisions = make_int2(1,1);

  leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_height = 0.7;

  trunk_radius = 0.05;

  cordon_height = 0.9;

  cordon_radius = 0.02;

  shoot_length = 0.9;

  shoot_radius = 0.0025;

  shoots_per_cordon = 10;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

  cluster_height_max = 0.15;

  grape_color = make_RGBcolor(0.18,0.2,0.25);

  grape_subdivisions = 8;

  plant_spacing = 2;

  row_spacing = 2;

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;
  
}

SplitGrapevineParameters::SplitGrapevineParameters(void){

  leaf_width = 0.18;

  leaf_subdivisions = make_int2(1,1);

  leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_height = 1.3;

  trunk_radius = 0.05;

  cordon_height = 1.5;

  cordon_radius = 0.02;

  cordon_spacing = 1.f;

  shoot_length = 1.2;

  shoot_radius = 0.0025;

  shoots_per_cordon = 10;

  shoot_angle_tip = 0.4*M_PI;

  shoot_angle_base = 0.;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

  cluster_height_max = 0.1;

  grape_color = make_RGBcolor(0.18,0.2,0.25);

  grape_subdivisions = 8;
  
  plant_spacing = 2;

  row_spacing = 4;

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;
  
}

UnilateralGrapevineParameters::UnilateralGrapevineParameters(void){

  leaf_width = 0.18;

  leaf_subdivisions = make_int2(1,1);

  leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_height = 0.7;

  trunk_radius = 0.05;

  cordon_height = 0.9;

  cordon_radius = 0.04;

  shoot_length = 0.9;

  shoot_radius = 0.0025;

  shoots_per_cordon = 20;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

  cluster_height_max = 0.1;

  grape_color = make_RGBcolor(0.18,0.2,0.25);

  grape_subdivisions = 8;

  plant_spacing = 1.5;

  row_spacing = 2;

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;
  
}

GobletGrapevineParameters::GobletGrapevineParameters(void){

  leaf_width = 0.18;

  leaf_subdivisions = make_int2(1,1);

  leaf_texture_file = "plugins/canopygenerator/textures/GrapeLeaf.png";

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_height = 0.7;

  trunk_radius = 0.05;

  cordon_height = 0.9;

  cordon_radius = 0.02;

  shoot_length = 0.9;

  shoot_radius = 0.0025;

  shoots_per_cordon = 10;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

  cluster_height_max = 0.1;

  grape_color = make_RGBcolor(0.18,0.2,0.25);

  grape_subdivisions = 8;

  plant_spacing = 2;

  row_spacing = 2;

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;  
  
}

WhiteSpruceCanopyParameters::WhiteSpruceCanopyParameters(void){

  needle_width = 0.0005;

  needle_length = 0.05;

  needle_color = make_RGBcolor(0.67,0.9,0.56);

  needle_subdivisions = make_int2(1,1);

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_height = 10;

  trunk_radius = 0.15;

  base_height = 2.0;

  crown_radius = 0.65;

  shoot_radius = 0.02;

  level_spacing = 0.35;

  branches_per_level = 8;

  shoot_angle = 0.3*M_PI;

  canopy_configuration = "random";

  plant_spacing = make_vec2(10,10);

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;

}

TomatoParameters::TomatoParameters(void){

  leaf_length = 0.2;

  leaf_subdivisions = make_int2(1,1);

  leaf_texture_file = "plugins/canopygenerator/textures/TomatoLeaf_big.png";

  shoot_color = make_RGBcolor(0.35,0.45,0.2);

  shoot_subdivisions = 10;

  plant_height = 1.;

  fruit_radius = 0.03;

  fruit_color = make_RGBcolor(0.7,0.28,0.2);

  fruit_subdivisions = 8;

  plant_spacing = 2;

  row_spacing = 2;

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;
  
}

StrawberryParameters::StrawberryParameters(void){

  leaf_length = 0.1;

  leaf_subdivisions = make_int2(4,4);

  leaf_texture_file = "plugins/canopygenerator/textures/StrawberryLeaf.png";

  stem_color = make_RGBcolor(0.35,0.45,0.2);

  stem_subdivisions = 10;

  stems_per_plant = 50;
  
  stem_radius = 0.005;

  plant_height = 0.4;

  fruit_radius = 0.025;

  fruit_texture_file = "plugins/canopygenerator/textures/StrawberryTexture.jpg";

  fruit_subdivisions = 12;

  clusters_per_stem = 0.6;

  plant_spacing = 0.5;

  row_spacing = 1.5;

  plant_count = make_int2(4,2);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;
  
}

WalnutCanopyParameters::WalnutCanopyParameters(void){

  leaf_length = 0.15;

  leaf_subdivisions = make_int2(1,2);

  leaf_texture_file = "plugins/canopygenerator/textures/StrawberryLeaf.png";

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 15;

  stems_per_plant = 50;
  
  trunk_radius = 0.15;

  trunk_height = 4.f;

  tree_height = 6;

  branch_length = make_vec3(4,0.75,0.75);

  crown_radius = 2.5;

  fruit_radius = 0.025;

  fruit_texture_file = "plugins/canopygenerator/textures/StrawberryTexture.jpg";

  fruit_subdivisions = 12;

  clusters_per_stem = 0.6;

  plant_spacing = 6;

  row_spacing = 8;

  plant_count = make_int2(4,2);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;
  
}

CanopyGenerator::CanopyGenerator( helios::Context* __context ){

  context = __context;

  //seed the random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);

  printmessages = true;

}

int CanopyGenerator::selfTest( void ){

  std::cout << "Running canopy generator plug-in self-test..." << std::endl;

  Context context;

  std::cout << "Generating default homogeneous canopy..." << std::flush;

  CanopyGenerator canopygenerator_0(&context);
  canopygenerator_0.disableMessages();
  HomogeneousCanopyParameters params_0;
  canopygenerator_0.buildCanopy(params_0);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default spherical crowns canopy..." << std::flush;

  CanopyGenerator canopygenerator_1(&context);
  canopygenerator_1.disableMessages();
  SphericalCrownsCanopyParameters params_1;
  canopygenerator_1.buildCanopy(params_1);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default VSP grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_2(&context);
  canopygenerator_2.disableMessages();
  VSPGrapevineParameters params_2;
  canopygenerator_2.buildCanopy(params_2);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default split trellis grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_3(&context);
  canopygenerator_3.disableMessages();
  SplitGrapevineParameters params_3;
  canopygenerator_3.buildCanopy(params_3);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default unilateral trellis grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_4(&context);
  canopygenerator_4.disableMessages();
  UnilateralGrapevineParameters params_4;
  canopygenerator_4.buildCanopy(params_4);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default goblet trellis grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_5(&context);
  canopygenerator_5.disableMessages();
  GobletGrapevineParameters params_5;
  canopygenerator_5.buildCanopy(params_5);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default strawberry canopy..." << std::flush;

  CanopyGenerator canopygenerator_7(&context);
  canopygenerator_7.disableMessages();
  StrawberryParameters params_7;
  canopygenerator_7.buildCanopy(params_7);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating homogeneous canopy with randomly deleted primitives..." << std::flush;

  CanopyGenerator canopygenerator_6(&context);
  canopygenerator_6.disableMessages();
  HomogeneousCanopyParameters params_6;
  canopygenerator_6.buildCanopy(params_6);

  std::vector<uint> UUIDs_leaves = flatten( canopygenerator_6.getLeafUUIDs(0) );

  context.deletePrimitive(UUIDs_leaves.at(0));
  context.deletePrimitive(UUIDs_leaves.at(11));
  context.deletePrimitive(UUIDs_leaves.at(23));
  context.deletePrimitive(UUIDs_leaves.back());

  UUIDs_leaves = flatten( canopygenerator_6.getLeafUUIDs(0) );

  bool fail_flag = false;
  for( uint p=0; p<UUIDs_leaves.size(); p++ ){
    if( !context.doesPrimitiveExist( UUIDs_leaves.at(p) ) ){
      fail_flag = true;
    }
  }
  
  std::vector<uint> UUIDs_all = canopygenerator_6.getAllUUIDs(0);

  fail_flag = false;
  for( uint p=0; p<UUIDs_all.size(); p++ ){
    if( !context.doesPrimitiveExist( UUIDs_all.at(p) ) ){
      fail_flag = true;
    }
  }
  
  context.deletePrimitive( context.getAllUUIDs() );

  std::cout << "done." << std::endl;
  
  if( fail_flag ){
    std::cout << "failed." << std::endl;
    return 1;
  }else{    
    std::cout << "passed." << std::endl;
    return 0;
  }
  
}

void CanopyGenerator::buildGround( const helios::vec3 ground_origin, const helios::vec2 ground_extent, const helios::int2 texture_subtiles, const helios::int2 texture_subpatches, const char* ground_texture_file  ){
  buildGround( ground_origin, ground_extent, texture_subtiles, texture_subpatches, ground_texture_file, 0.f );
}

void CanopyGenerator::buildGround( const helios::vec3 ground_origin, const helios::vec2 ground_extent, const helios::int2 texture_subtiles, const helios::int2 texture_subpatches, const char* ground_texture_file, const float ground_rotation  ){

  if( printmessages ){
    std::cout << "Ground geometry..." << std::flush;
  }

  vec2 dx_tile( ground_extent.x/float(texture_subtiles.x), ground_extent.y/float(texture_subtiles.y) );

  vec2 dx_subpatch( dx_tile.x/float(texture_subpatches.x), dx_tile.y/float(texture_subpatches.y) );

  std::vector<uint> UUIDs;
  for( int j=0; j<texture_subtiles.y; j++ ){
    for( int i=0; i<texture_subtiles.x; i++ ){

      vec3 center = ground_origin + make_vec3( -0.5*ground_extent.x+(i+0.5)*dx_tile.x, -0.5*ground_extent.y+(j+0.5)*dx_tile.y, 0 );

      if( ground_rotation!=0 ){
	center = rotatePointAboutLine( center, ground_origin, make_vec3(0,0,1), ground_rotation );
      }
      
      UUIDs = context->addTile( center, dx_tile, make_SphericalCoord(0,-ground_rotation), texture_subpatches, ground_texture_file );

      UUID_ground.insert( UUID_ground.begin(), UUIDs.begin(), UUIDs.end() );
      
    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Ground consists of " << UUID_ground.size() << " total primitives." << std::endl;
  }
  
}

void CanopyGenerator::buildCanopy( const HomogeneousCanopyParameters params ){

  if( printmessages ){
    std::cout << "Building homogeneous canopy..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  UUID_leaf.resize(1);

  float solidFractionx;
  if(params.leaf_texture_file.size()==0){
    solidFractionx = 1.0;
  }else{
    helios::Texture texture(params.leaf_texture_file.c_str());
    solidFractionx = texture.getSolidFraction();
  }

  float leafArea = params.leaf_size.x*params.leaf_size.y*solidFractionx;
  int Nleaves = round(params.leaf_area_index*params.canopy_extent.x*params.canopy_extent.y/leafArea);

  float Lmax = fmax(params.leaf_size.x,params.leaf_size.y);

  uint ID0;
  if( params.leaf_texture_file.size()==0 ){
    ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_color );
  }else{
    ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() );
  }

  for( int i=0; i<Nleaves; i++ ){

    float rx = unif_distribution(generator);
    float ry = unif_distribution(generator);
    float rz = unif_distribution(generator);

    float rp = unif_distribution(generator);

    vec3 position = params.canopy_origin + make_vec3( (-0.5+rx)*params.canopy_extent.x, (-0.5+ry)*params.canopy_extent.y, 0.5*Lmax+rz*(params.canopy_height-Lmax) );

    SphericalCoord rotation( 1.f, sampleLeafPDF(params.leaf_angle_distribution.c_str()), 2.f*M_PI*rp );

    uint ID = context->copyObject(ID0);
    context->getObjectPointer(ID)->rotate(-rotation.elevation,"y");
    context->getObjectPointer(ID)->rotate(rotation.azimuth,"z");
    context->getObjectPointer(ID)->translate(position);

    std::vector<uint> UUID = context->getObjectPointer(ID)->getPrimitiveUUIDs();

    UUID_leaf.front().push_back(UUID);

  }

  context->deleteObject(ID0);


  if( printmessages ){
    std::cout << "done." << std::endl;
    std::vector<uint> UUIDs_all = getAllUUIDs(0);
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << UUIDs_all.size() << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const SphericalCrownsCanopyParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of spherical crowns..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  float r = params.crown_radius;

  float solidFractionx;
  if(params.leaf_texture_file.size()==0){
    solidFractionx = 1.0;
  }else{
    helios::Texture texture(params.leaf_texture_file.c_str());
    solidFractionx = texture.getSolidFraction();
  }

  float leafArea = params.leaf_size.x*params.leaf_size.y*solidFractionx;
  int Nleaves = round(4.f/3.f*M_PI*r*r*r*params.leaf_area_density/leafArea);

  vec2 canopy_extent( params.plant_spacing.x*float(params.plant_count.x), params.plant_spacing.y*float(params.plant_count.y) );

  std::string cconfig = params.canopy_configuration;
  if( cconfig.compare("uniform")!=0 && cconfig.compare("random")!=0 ){
    std::cout << "WARNING: Unknown canopy configuration parameter for spherical crowns canopy: " << cconfig << ". Using uniformly spaced configuration." << std::endl;
    cconfig = "uniform";
  }

  UUID_leaf.resize(params.plant_count.x*params.plant_count.y);

  uint ID0;
  if( params.leaf_texture_file.size()==0 ){
    ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_color );
  }else{
    ID0 = context->addTileObject( make_vec3(0,0,0), params.leaf_size, make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() );
  }

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center;
      if( cconfig.compare("uniform")==0 ){
	center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing.x, -0.5*canopy_extent.y+(j+0.5)*params.plant_spacing.y, r );
      }else if( cconfig.compare("random")==0 ){
	float rx = unif_distribution(generator);
	float ry = unif_distribution(generator);
	center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+i*params.plant_spacing.x+r+(params.plant_spacing.x-2.f*r)*rx, -0.5*canopy_extent.y+j*params.plant_spacing.y+r+(params.plant_spacing.y-2.f*r)*ry, r );
      }

      if( params.canopy_rotation!=0 ){
	center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
      }
	
      for (int l=0; l<Nleaves; l++ ){
	  
	float u = unif_distribution(generator);
	float v = unif_distribution(generator);
	float theta = u * 2.0 * M_PI;
	float phi = acosf(2.0 * v - 1.0);
	float rad = pow(unif_distribution(generator),1.f/3.f);
	float x = r * rad * sinf(phi) * cosf(theta);
	float y = r * rad * sinf(phi) * sinf(theta);
	float z = r * rad * cosf(phi);

	theta = sampleLeafPDF(params.leaf_angle_distribution.c_str());
	phi = 2.f*M_PI*unif_distribution(generator);

	uint ID = context->copyObject(ID0);
	context->getObjectPointer(ID)->rotate(-theta,"y");
	context->getObjectPointer(ID)->rotate(phi,"z");
	context->getObjectPointer(ID)->translate(center+make_vec3(x,y,z));

	std::vector<uint> UUID = context->getObjectPointer(ID)->getPrimitiveUUIDs();

	UUID_leaf.at(plant_ID).push_back(UUID);

      }

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;
      
    }
  }

  context->deleteObject(ID0);

  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const VSPGrapevineParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of VSP grapevine..." << std::flush;
  }

  if( params.cordon_height<params.trunk_height ){
    std::cout << "failed." << std::endl;
    std::cerr << "ERROR: Cannot build VSP grapevine canopy. Cordon height cannot be less than the trunk height." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      grapevineVSP( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }


  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const SplitGrapevineParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of split trellis grapevine..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      grapevineSplit( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }


  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const UnilateralGrapevineParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of unilateral trellis grapevine..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      grapevineUnilateral( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }


  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const GobletGrapevineParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of goblet trellis grapevine..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      grapevineGoblet( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }


  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const WhiteSpruceCanopyParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of white spruce trees..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing.x*float(params.plant_count.x), params.plant_spacing.y*float(params.plant_count.y) );

  std::string cconfig = params.canopy_configuration;
  if( cconfig.compare("uniform")!=0 && cconfig.compare("random")!=0 ){
    std::cout << "WARNING: Unknown canopy configuration parameter for white spruce canopy: " << cconfig << ". Using uniformly spaced configuration." << std::endl;
    cconfig = "uniform";
  }

  float r = params.crown_radius;

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center;
      if( cconfig.compare("uniform")==0 ){
	center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing.x, -0.5*canopy_extent.y+(j+0.5)*params.plant_spacing.y, 0 );
      }else if( cconfig.compare("random")==0 ){
	float rx = unif_distribution(generator);
	float ry = unif_distribution(generator);
	center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+i*params.plant_spacing.x+r+(params.plant_spacing.x-2.f*r)*rx, -0.5*canopy_extent.y+j*params.plant_spacing.y+r+(params.plant_spacing.y-2.f*r)*ry, 0 );
      }

      if( params.canopy_rotation!=0 ){
	center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
      }
	
       whitespruce( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;
      
    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
    std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const TomatoParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of tomato plants..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      tomato( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
    //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const StrawberryParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of strawberry plants..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      strawberry( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
    //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

void CanopyGenerator::buildCanopy( const WalnutCanopyParameters params ){

  if( printmessages ){
    std::cout << "Building canopy of walnut trees..." << std::flush;
  }

  std::uniform_real_distribution<float> unif_distribution;

  vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );

  uint plant_ID = 0;
  uint prim_count = 0;
  for( int j=0; j<params.plant_count.y; j++ ){
    for( int i=0; i<params.plant_count.x; i++ ){

      vec3 center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing, -0.5*canopy_extent.y+(j+0.5)*params.row_spacing, 0 );

      walnut( params, center );

      std::vector<uint> UUIDs_all = getAllUUIDs(plant_ID);
      prim_count += UUIDs_all.size();

      plant_ID++;

    }
  }

  if( printmessages ){
    std::cout << "done." << std::endl;
    //std::cout << "Canopy consists of " << UUID_leaf.size()*UUID_leaf.front().size() << " leaves and " << prim_count << " total primitives." << std::endl;
  }

}

float getVariation( float V, std::minstd_rand0& generator ){

  std::uniform_real_distribution<float> unif_distribution;

  return -V + 2.f*unif_distribution(generator)*V;
  
}

float CanopyGenerator::sampleLeafAngle( const std::vector<float> leafAngleDist ){

  std::vector<float> gL = leafAngleDist;
  
  float dTheta = 0.5f*M_PI/float(gL.size());

  //make sure PDF is properly normalized
  float norm = 0;
  for( int i=0; i<gL.size(); i++ ){
    norm += gL.at(i)*dTheta;
  }
  for( int i=0; i<gL.size(); i++ ){
    gL.at(i) /= norm;
  }
  norm = 0;
  for( int i=0; i<gL.size(); i++ ){
    norm += gL.at(i)*dTheta;
  }
  assert( fabs(norm-1)<0.001 );
  
  //calculate the leaf angle CDF
  std::vector<float> leafAngleCDF;
  leafAngleCDF.resize( gL.size() );
  float tsum = 0;
  for( int i=0; i<gL.size(); i++ ){
    tsum += gL.at(i)*dTheta;
    leafAngleCDF.at(i) = tsum;
  }

  assert( fabs(tsum-1.f)<0.001 );
  
  //draw from leaf angle PDF
  std::uniform_real_distribution<float> unif_distribution;
  float rt = unif_distribution(generator);

  float theta = -1;
  for( int i=0; i<gL.size(); i++ ){
    if( rt<leafAngleCDF.at(i) ){
      theta = (i+unif_distribution(generator))*dTheta;
      break;
    }
  }

  assert( theta!=-1 );
  
  return theta;
  
}

std::vector<uint> CanopyGenerator::getTrunkUUIDs( const uint PlantID ){
  if( PlantID>=UUID_trunk.size() ){
    std::cerr << "ERROR (CanopyGenerator::getTrunkUUIDs): Cannot get UUIDs for plant " << PlantID << " because only " << UUID_trunk.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<uint> UUID;

  for( size_t i=0; i<UUID_trunk.at(PlantID).size(); i++ ){
 
    if( context->doesPrimitiveExist(UUID_trunk.at(PlantID).at(i)) ){
      UUID.push_back(UUID_trunk.at(PlantID).at(i));
    }
          
  }
  
  return UUID;
  
}

std::vector<uint> CanopyGenerator::getTrunkUUIDs( void ){
  return flatten( UUID_trunk );
}

std::vector<uint> CanopyGenerator::getBranchUUIDs( const uint PlantID ){
  if( PlantID>=UUID_branch.size() ){
    std::cerr << "ERROR (CanopyGenerator::getBranchUUIDs): Cannot get UUIDs for plant " << PlantID << " because only " << UUID_branch.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<uint> UUID;

  for( size_t i=0; i<UUID_branch.at(PlantID).size(); i++ ){
 
    if( context->doesPrimitiveExist(UUID_branch.at(PlantID).at(i)) ){
      UUID.push_back(UUID_branch.at(PlantID).at(i));
    }
          
  }
  
  return UUID;

}

std::vector<uint> CanopyGenerator::getBranchUUIDs( void ){
  return flatten( UUID_branch );
}

std::vector<std::vector<uint> > CanopyGenerator::getLeafUUIDs( const uint PlantID ){
  if( PlantID>=UUID_leaf.size() ){
    std::cerr << "ERROR (CanopyGenerator::getLeafUUIDs): Cannot get UUIDs for plant " << PlantID << " because only " << UUID_leaf.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }

  //in case primitives have been deleted, only return UUIDs that still exist
  
  std::vector<std::vector<uint> > UUID;

  for( size_t j=0; j<UUID_leaf.at(PlantID).size(); j++ ){
    std::vector<uint> U;
    for( size_t i=0; i<UUID_leaf.at(PlantID).at(j).size(); i++ ){
      
      if( context->doesPrimitiveExist(UUID_leaf.at(PlantID).at(j).at(i)) ){
	U.push_back(UUID_leaf.at(PlantID).at(j).at(i));
      }
      
    }

    if( U.size()>0 ){
      UUID.push_back(U);
    }
    
  }
  
  return UUID;
}

std::vector<uint> CanopyGenerator::getLeafUUIDs( void ){
  return flatten( UUID_leaf );
}

std::vector<std::vector<std::vector<uint> > > CanopyGenerator::getFruitUUIDs( const uint PlantID ){
  if( PlantID>=UUID_fruit.size() ){
    std::cerr << "ERROR (CanopyGenerator::getFruitUUIDs): Cannot get UUIDs for plant " << PlantID << " because only " << UUID_fruit.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }

  std::vector<std::vector<std::vector<uint> > > UUID;

  for( size_t k=0; k<UUID_fruit.at(PlantID).size(); k++ ){
    std::vector<std::vector<uint> > U2;
    for( size_t j=0; j<UUID_fruit.at(PlantID).at(k).size(); j++ ){
      std::vector<uint> U1;
      for( size_t i=0; i<UUID_fruit.at(PlantID).at(k).at(j).size(); i++ ){

	if( context->doesPrimitiveExist(UUID_fruit.at(PlantID).at(k).at(j).at(i)) ){
	  U1.push_back(UUID_fruit.at(PlantID).at(k).at(j).at(i));
	}
      
      }

      if( U1.size()>0 ){
	U2.push_back(U1);
      }

    }

    if( U2.size()>0 ){
      UUID.push_back(U2);
    }
    
  }
  
  return UUID;

}

std::vector<uint> CanopyGenerator::getFruitUUIDs( void ){

  std::vector<uint> UUIDs, U;

  for( uint p=0; p<UUID_fruit.size(); p++ ){
    U = flatten( UUID_fruit.at(p) );
    UUIDs.insert( UUIDs.end(), U.begin(), U.end() );
  }
  
  return UUIDs;
}

std::vector<uint> CanopyGenerator::getGroundUUIDs( void ){

  std::vector<uint> UUID;

  for( size_t i=0; i<UUID_ground.size(); i++ ){
 
    if( context->doesPrimitiveExist(UUID_ground.at(i)) ){
      UUID.push_back(UUID_ground.at(i));
    }
          
  }
  
  return UUID;
  
}

std::vector<uint> CanopyGenerator::getAllUUIDs( const uint PlantID ){
  std::vector<uint> UUIDs;
  if( UUID_trunk.size()>PlantID ){
    UUIDs.insert(UUIDs.end(),UUID_trunk.at(PlantID).begin(),UUID_trunk.at(PlantID).end());
  }
  if( UUID_branch.size()>PlantID ){
    UUIDs.insert(UUIDs.end(),UUID_branch.at(PlantID).begin(),UUID_branch.at(PlantID).end());
  }
  if( UUID_leaf.size()>PlantID ){
    for( int i=0; i<UUID_leaf.at(PlantID).size(); i++ ){
      UUIDs.insert(UUIDs.end(),UUID_leaf.at(PlantID).at(i).begin(),UUID_leaf.at(PlantID).at(i).end());
    }
  }
  if( UUID_fruit.size()>PlantID ){
    for( int j=0; j<UUID_fruit.at(PlantID).size(); j++ ){
      for( int i=0; i<UUID_fruit.at(PlantID).at(j).size(); i++ ){
	UUIDs.insert(UUIDs.end(),UUID_fruit.at(PlantID).at(j).at(i).begin(),UUID_fruit.at(PlantID).at(j).at(i).end());
      }
    }
  }

  std::vector<uint> U;

  for( size_t p=0; p<UUIDs.size(); p++ ){
    if( context->doesPrimitiveExist(UUIDs.at(p)) ){
      U.push_back(UUIDs.at(p));
    }
  }
  
  return U;
}

uint CanopyGenerator::getPlantCount( void ){
  return UUID_leaf.size();
}

void CanopyGenerator::seedRandomGenerator( const uint seed ){
  generator.seed(seed);
}

void CanopyGenerator::disableMessages( void ){
  printmessages=false;
}

void CanopyGenerator::enableMessages( void ){
  printmessages=true;
}

helios::vec3 interpolateTube( const std::vector<helios::vec3> P, const float frac ){

  assert( frac>=0 && frac<=1 );
  assert( P.size()>0 );

  float dl=0.f;
  for( int i=0; i<P.size()-1; i++ ){
    dl+=(P.at(i+1)-P.at(i)).magnitude();
  }

  float f = 0;
  for( int i=0; i<P.size()-1; i++ ){

    float dseg = (P.at(i+1)-P.at(i)).magnitude();

    float fplus = f+dseg/dl;

    if( fplus>=1.f ){
      fplus = 1.f+1e-3;
    }

    if( frac>=f && (frac<=fplus || fabs(frac-fplus)<0.0001 ) ){

      vec3 V = P.at(i) + (frac-f)/(fplus-f)*(P.at(i+1)-P.at(i));

      return V;
    }

    f=fplus;
  }

  return P.front();

}

float interpolateTube( const std::vector<float> P, const float frac ){

  assert( frac>=0 && frac<=1 );
  assert( P.size()>0 );

  float dl=0.f;
  for( int i=0; i<P.size()-1; i++ ){
    dl+=(P.at(i+1)-P.at(i));
  }

  float f = 0;
  for( int i=0; i<P.size()-1; i++ ){

    float dseg = (P.at(i+1)-P.at(i));

    float fplus = f+dseg/dl;

    if( fplus>=1.f ){
      fplus = 1.f+1e-3;
    }

    if( frac>=f && (frac<=fplus || fabs(frac-fplus)<0.0001 ) ){

      float V = P.at(i) + (frac-f)/(fplus-f)*(P.at(i+1)-P.at(i));
      
      return V;
    }

    f=fplus;
  }

  return P.front();

}

float evaluateCDFresid( const char* distribution, float thetaL, float ru ){

  assert( ru>=0.f && ru<=1.f );

  float CDFresid;
  if( strcmp(distribution,"planophile")==0 ){
    CDFresid = ru - 2.f/M_PI*(thetaL+0.5*sinf(2.f*thetaL));
  }else if( strcmp(distribution,"erectophile")==0 ){
    CDFresid = ru - 2.f/M_PI*(thetaL-0.5*sinf(2.f*thetaL));
  }else if( strcmp(distribution,"plagiophile")==0 ){
    CDFresid = ru - 2.f/M_PI*(thetaL-0.25*sinf(4.f*thetaL));
  }else if( strcmp(distribution,"extremophile")==0 ){
    CDFresid = ru - 2.f/M_PI*(thetaL+0.25*sinf(4.f*thetaL));
  }else{

  }

  return CDFresid;

}

float CanopyGenerator::sampleLeafPDF( const char* distribution ){

  float thetaL;

  std::uniform_real_distribution<float> unif_distribution;
  
  float ru = unif_distribution(generator);

  if( strcmp(distribution,"spherical")==0 ){
    thetaL = acos_safe(1.f-ru);
  }else if( strcmp(distribution,"uniform")==0 ){
    thetaL = ru*0.5*M_PI;
  }else if( strcmp(distribution,"planophile")==0 || strcmp(distribution,"erectophile")==0 || strcmp(distribution,"plagiophile")==0 || strcmp(distribution,"extremophile")==0  ){

    float err_max = 0.0001;
    uint max_iter = 100;

    float tL_old_old = 0.25*M_PI;;

    float tL_old = 0.24*M_PI;

    float resid_old = evaluateCDFresid(distribution,tL_old,ru);
    float resid_old_old = evaluateCDFresid(distribution,tL_old_old,ru);
  
    float resid = 100;
    float err = resid;
    uint iter = 0;
    while( err>err_max && iter<max_iter ){
      
      if( resid_old==resid_old_old ){//this condition will cause NaN
	err=0;
	break;
      }
      
      thetaL = fabs((tL_old_old*resid_old-tL_old*resid_old_old)/(resid_old-resid_old_old));
      
      resid = evaluateCDFresid(distribution,thetaL,ru);
      
      resid_old_old = resid_old;
      resid_old = resid;
      
      err = fabs(tL_old-tL_old_old)/fabs(tL_old_old);
      
      tL_old_old = tL_old;
      tL_old = thetaL;
      
      iter++;

    }

    assert( err<=err_max );

  }else{
    std::cerr << "ERROR (sampleLeafPDF): Invalid leaf angle distribution of " << distribution << " specified." << std::endl;
    throw 1;
  }

  return thetaL;
  
}
