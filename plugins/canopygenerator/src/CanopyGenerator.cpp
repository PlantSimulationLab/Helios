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

  leaf_area_index = 1.f;

  canopy_height = 1.f;

  canopy_extent = make_vec2(5,5);

  canopy_origin = make_vec3(0,0,0);
  
}

SphericalCrownsCanopyParameters::SphericalCrownsCanopyParameters(void){

  leaf_size = make_vec2(0.025,0.025);

  leaf_subdivisions = make_int2(1,1);

  leaf_color = RGB::green;

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

  shoot_radius = 0.005;

  shoots_per_cordon = 10;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

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

  shoot_radius = 0.005;

  shoots_per_cordon = 10;

  shoot_angle_tip = 0.4*M_PI;

  shoot_angle_base = 0.;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

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

  shoot_radius = 0.005;

  shoots_per_cordon = 20;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

  plant_spacing = 2;

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

  shoot_radius = 0.005;

  shoots_per_cordon = 10;

  leaf_spacing_fraction = 0.6;

  grape_radius = 0.0075;

  cluster_radius = 0.03;

  plant_spacing = 2;

  row_spacing = 2;

  plant_count = make_int2(3,3);

  canopy_origin = make_vec3(0,0,0);

  canopy_rotation = 0;  
  
}

WhiteSpruceCanopyParameters::WhiteSpruceCanopyParameters(void){

  needle_width = 0.001;

  needle_length = 0.05;

  needle_color = RGB::forestgreen;

  needle_subdivisions = make_int2(1,1);

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_height = 10;

  trunk_radius = 0.1;

  base_height = 1.25;

  crown_radius = 1.;

  shoot_radius = 0.02;

  level_spacing = 0.25;

  branches_per_level = 10;

  shoot_angle = 0.3*M_PI;

  canopy_configuration = "random";

  plant_spacing = make_vec2(10,10);

  plant_count = make_int2(3,3);

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
  HomogeneousCanopyParameters params_0;
  canopygenerator_0.buildCanopy(params_0);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default spherical crowns canopy..." << std::flush;

  CanopyGenerator canopygenerator_1(&context);
  SphericalCrownsCanopyParameters params_1;
  canopygenerator_1.buildCanopy(params_1);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default VSP grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_2(&context);
  VSPGrapevineParameters params_2;
  canopygenerator_2.buildCanopy(params_2);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default split trellis grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_3(&context);
  SplitGrapevineParameters params_3;
  canopygenerator_3.buildCanopy(params_3);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default unilateral trellis grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_4(&context);
  UnilateralGrapevineParameters params_4;
  canopygenerator_4.buildCanopy(params_4);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "Generating default goblet trellis grapevine canopy..." << std::flush;

  CanopyGenerator canopygenerator_5(&context);
  GobletGrapevineParameters params_5;
  canopygenerator_5.buildCanopy(params_5);
  context.deletePrimitive( context.getAllUUIDs() );
  
  std::cout << "done." << std::endl;

  std::cout << "passed." << std::endl;
  return 0;
  
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

  int Nleaves = round(params.leaf_area_index*params.canopy_extent.x*params.canopy_extent.y/params.leaf_size.x/params.leaf_size.y);

  float Lmax = fmax(params.leaf_size.x,params.leaf_size.y);

  for( int i=0; i<Nleaves; i++ ){

    float rx = unif_distribution(generator);
    float ry = unif_distribution(generator);
    float rz = unif_distribution(generator);

    float rt = unif_distribution(generator);
    float rp = unif_distribution(generator);

    vec3 position = params.canopy_origin + make_vec3( (-0.5+rx)*params.canopy_extent.x, (-0.5+ry)*params.canopy_extent.y, 0.5*Lmax+rz*(params.canopy_height-Lmax) );

    SphericalCoord rotation( 1.f, acos(1.f-rt), 2.f*M_PI*rp );

    std::vector<uint> UUID;
    if( params.leaf_texture_file.size()==0 ){
      UUID = context->addTile( position, params.leaf_size, rotation, params.leaf_subdivisions, params.leaf_color );
    }else{
      UUID = context->addTile( position, params.leaf_size, rotation, params.leaf_subdivisions, params.leaf_texture_file.c_str() );
    }

    UUID_leaf.front().push_back(UUID);

  }


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
  
  int Nleaves = round(4.f/3.f*M_PI*r*r*r*params.leaf_area_density/params.leaf_size.x/params.leaf_size.y);

  vec2 canopy_extent( params.plant_spacing.x*float(params.plant_count.x), params.plant_spacing.y*float(params.plant_count.y) );

  std::string cconfig = params.canopy_configuration;
  if( cconfig.compare("uniform")!=0 && cconfig.compare("random")!=0 ){
    std::cout << "WARNING: Unknown canopy configuration parameter for spherical crowns canopy: " << cconfig << ". Using uniformly spaced configuration." << std::endl;
    cconfig = "uniform";
  }

  UUID_leaf.resize(params.plant_count.x*params.plant_count.y);

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
	      
	phi = 2.f*M_PI*unif_distribution(generator);

	std::vector<uint> UUID;
	if( params.leaf_texture_file.size()==0 ){
	  UUID=context->addTile( center+make_vec3(x,y,z), params.leaf_size, make_SphericalCoord(theta,phi), params.leaf_subdivisions, params.leaf_color );
	}else{
	  UUID=context->addTile( center+make_vec3(x,y,z), params.leaf_size, make_SphericalCoord(theta,phi), params.leaf_subdivisions, params.leaf_texture_file.c_str() );
	}

	UUID_leaf.at(plant_ID).push_back(UUID);

      }

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

std::vector<uint> CanopyGenerator::getTrunkUUIDs( const uint TreeID ){
  if( TreeID>=UUID_trunk.size() ){
    std::cerr << "ERROR (CanopyGenerator::getTrunkUUIDs): Cannot get UUIDs for plant " << TreeID << " because only " << UUID_trunk.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }
  return UUID_trunk.at(TreeID);
}

std::vector<uint> CanopyGenerator::getBranchUUIDs( const uint TreeID ){
  if( TreeID>=UUID_trunk.size() ){
    std::cerr << "ERROR (CanopyGenerator::getBranchUUIDs): Cannot get UUIDs for plant " << TreeID << " because only " << UUID_branch.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }
  return UUID_branch.at(TreeID);
}

std::vector<std::vector<uint> > CanopyGenerator::getLeafUUIDs( const uint TreeID ){
  if( TreeID>=UUID_trunk.size() ){
    std::cerr << "ERROR (CanopyGenerator::getLeafUUIDs): Cannot get UUIDs for plant " << TreeID << " because only " << UUID_leaf.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }
  return UUID_leaf.at(TreeID);
}

std::vector<std::vector<std::vector<uint> > > CanopyGenerator::getFruitUUIDs( const uint TreeID ){
  if( TreeID>=UUID_trunk.size() ){
    std::cerr << "ERROR (CanopyGenerator::getFruitUUIDs): Cannot get UUIDs for plant " << TreeID << " because only " << UUID_fruit.size() << " plants have been built." << std::endl;
    exit(EXIT_FAILURE);
  }
  return UUID_fruit.at(TreeID);
}

std::vector<uint> CanopyGenerator::getAllUUIDs( const uint TreeID ){
  std::vector<uint> UUIDs;
  if( UUID_trunk.size()>TreeID ){
    UUIDs.insert(UUIDs.end(),UUID_trunk.at(TreeID).begin(),UUID_trunk.at(TreeID).end());
  }
  if( UUID_branch.size()>TreeID ){
    UUIDs.insert(UUIDs.end(),UUID_branch.at(TreeID).begin(),UUID_branch.at(TreeID).end());
  }
  if( UUID_leaf.size()>TreeID ){
    for( int i=0; i<UUID_leaf.at(TreeID).size(); i++ ){
      UUIDs.insert(UUIDs.end(),UUID_leaf.at(TreeID).at(i).begin(),UUID_leaf.at(TreeID).at(i).end());
    }
  }
  if( UUID_fruit.size()>TreeID ){
    for( int j=0; j<UUID_fruit.at(TreeID).size(); j++ ){
      for( int i=0; i<UUID_fruit.at(TreeID).at(j).size(); i++ ){
	UUIDs.insert(UUIDs.end(),UUID_fruit.at(TreeID).at(j).at(i).begin(),UUID_fruit.at(TreeID).at(j).at(i).end());
      }
    }
  }
  return UUIDs;
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
