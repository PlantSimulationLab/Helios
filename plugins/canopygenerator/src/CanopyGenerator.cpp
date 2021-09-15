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
#include "pugixml.hpp"

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

  buffer = "z";
  
}

SphericalCrownsCanopyParameters::SphericalCrownsCanopyParameters(void){

  leaf_size = make_vec2(0.025,0.025);

  leaf_subdivisions = make_int2(1,1);

  leaf_color = RGB::green;

  leaf_angle_distribution = "spherical";

  leaf_area_density = 1.f;

  crown_radius = make_vec3(0.5f,0.5f,0.5f);

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

  leaf_spacing_fraction = 0.5;

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

  leaf_texture_file = "plugins/canopygenerator/textures/WalnutLeaf.png";

  wood_texture_file = "plugins/canopygenerator/textures/wood.jpg";

  wood_subdivisions = 10;

  trunk_radius = 0.15;

  trunk_height = 4.f;

  branch_length = make_vec3(4,0.75,0.75);

  fruit_radius = 0.04;

  fruit_texture_file = "plugins/canopygenerator/textures/WalnutTexture.png";

  fruit_subdivisions = 16;

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

  std::cout << "Generating default walnut tree canopy..." << std::flush;

  CanopyGenerator canopygenerator_8(&context);
  canopygenerator_8.disableMessages();
  WalnutCanopyParameters params_8;
  canopygenerator_8.buildCanopy(params_8);
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

void CanopyGenerator::loadXML( const char* filename ){

  if( printmessages ){
    std::cout << "Reading XML file: " << filename << "..." << std::flush;
  }

  float nullvalue_f = 99999;
  int nullvalue_i = 99999;
  std::string nullvalue_s = "99999";
    
  //Check if file exists
  std::ifstream f(filename);
  if( !f.good() ){
    std::cerr << "failed.\n XML file does not exist." << std::endl;
    throw 1;
  }

  // Using "pugixml" parser.  See pugixml.org
  pugi::xml_document xmldoc;

  //load file
  pugi::xml_parse_result result = xmldoc.load_file(filename);

  //error checking
  if (!result){
    std::cout << "failed." << std::endl;
    std::cerr << "XML  file " << filename << " parsed with errors, attribute value: [" << xmldoc.child("node").attribute("attr").value() << "]\n";
    std::cerr << "Error description: " << result.description() << "\n";
    //cerr << "Error offset: " << result.offset << " (error at [..." << (filename + result.offset) << "]\n\n";
    throw 1;
  }

  pugi::xml_node helios = xmldoc.child("helios");

  if( helios.empty() ){
    std::cout << "failed." << std::endl;
    std::cerr << "ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags." << std::endl;
    throw 1;
  }

  //looping over any Canopy Generator blocks specified in XML file
  for (pugi::xml_node cgen = helios.child("canopygenerator"); cgen; cgen = cgen.next_sibling("CanopyGenerator")){

    //looping over any canopy types specified

    //Homogeneous Canopy
    for (pugi::xml_node s = cgen.child("HomogeneousCanopyParameters"); s; s = s.next_sibling("HomogeneousCanopyParameters")){

      HomogeneousCanopyParameters homogeneouscanopyparameters;

      // ----- leaf size ------//
      vec2 leaf_size = XMLloadvec2( s, "leaf_size");
      if( leaf_size.x != nullvalue_f && leaf_size.y != nullvalue_f ){
	homogeneouscanopyparameters.leaf_size = leaf_size;
      }

      // ----- leaf subdivisions ------//
      int2 leaf_subdivisions = XMLloadint2( s, "leaf_subdivisions");
      if( leaf_subdivisions.x != nullvalue_f && leaf_subdivisions.y != nullvalue_f ){
	homogeneouscanopyparameters.leaf_subdivisions = leaf_subdivisions;
      }

      // ----- leaf color ------//
      RGBcolor leaf_color = XMLloadrgb( s, "leaf_color");
      if( leaf_color.r != nullvalue_f && leaf_color.g != nullvalue_f && leaf_color.b != nullvalue_f ){
	homogeneouscanopyparameters.leaf_color = leaf_color;
      }

      // ----- leaf texture file ------//
      std::string leaf_texture_file = XMLloadstring( s, "leaf_texture_file");
      if( leaf_texture_file.compare(nullvalue_s)!=0 ){
	homogeneouscanopyparameters.leaf_texture_file = leaf_texture_file;
      }

      // ----- leaf area index ------//
      float LAI = XMLloadfloat( s, "leaf_area_index");
      if( LAI != nullvalue_f ){
	homogeneouscanopyparameters.leaf_area_index = LAI;
      }

      // ----- canopy height ------//
      float h = XMLloadfloat( s, "canopy_height" );
      if( h != nullvalue_f ){
	homogeneouscanopyparameters.canopy_height = h;
      }

      // ----- canopy extent ------//
      vec2 canopy_extent = XMLloadvec2( s, "canopy_extent");
      if( canopy_extent.x != nullvalue_f && canopy_extent.y != nullvalue_f ){
	homogeneouscanopyparameters.canopy_extent = canopy_extent;
      }

      // ----- canopy origin ------//
      vec3 canopy_origin = XMLloadvec3( s, "canopy_origin");
      if( canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f ){
	homogeneouscanopyparameters.canopy_origin = canopy_origin;
      }

      // ----- buffer ------//
      std::string buffer = XMLloadstring( s, "buffer");
      if( buffer.compare(nullvalue_s)!=0 ){
	homogeneouscanopyparameters.buffer = buffer;
      }

      buildCanopy( homogeneouscanopyparameters );
      
    }


    //VSP Grapevine Canopy
    for (pugi::xml_node s = cgen.child("VSPGrapevineParameters"); s; s = s.next_sibling("VSPGrapevineParameters")){

      VSPGrapevineParameters vspgrapevineparameters;

      float leaf_width = XMLloadfloat( s, "leaf_width");
      if( leaf_width != nullvalue_f ){
	vspgrapevineparameters.leaf_width = leaf_width;
      }

      int2 leaf_subdivisions = XMLloadint2( s, "leaf_subdivisions");
      if( leaf_subdivisions.x != nullvalue_f && leaf_subdivisions.y != nullvalue_f ){
	vspgrapevineparameters.leaf_subdivisions = leaf_subdivisions;
      }

      std::string leaf_texture_file = XMLloadstring( s, "leaf_texture_file");
      if( leaf_texture_file.compare(nullvalue_s)!=0 ){
	vspgrapevineparameters.leaf_texture_file = leaf_texture_file;
      }

      std::string wood_texture_file = XMLloadstring( s, "wood_texture_file");
      if( wood_texture_file.compare(nullvalue_s)!=0 ){
	vspgrapevineparameters.wood_texture_file = wood_texture_file;
      }

      int wood_subdivisions = XMLloadint( s, "wood_subdivisions");
      if( wood_subdivisions != nullvalue_i ){
	vspgrapevineparameters.wood_subdivisions = wood_subdivisions;
      }

      float h = XMLloadfloat( s, "trunk_height" );
      if( h != nullvalue_f ){
	vspgrapevineparameters.trunk_height = h;
      }

      float r = XMLloadfloat( s, "trunk_radius" );
      if( r != nullvalue_f ){
	vspgrapevineparameters.trunk_radius = r;
      }

      float ch = XMLloadfloat( s, "cordon_height" );
      if( ch != nullvalue_f ){
	vspgrapevineparameters.cordon_height = ch;
      }

      float cr = XMLloadfloat( s, "cordon_radius" );
      if( cr != nullvalue_f ){
	vspgrapevineparameters.cordon_radius = cr;
      }

      float sl = XMLloadfloat( s, "shoot_length" );
      if( sl != nullvalue_f ){
	vspgrapevineparameters.shoot_length = sl;
      }

      float sr = XMLloadfloat( s, "shoot_radius" );
      if( sr != nullvalue_f ){
	vspgrapevineparameters.shoot_radius = sr;
      }

      int spc = XMLloadint( s, "shoots_per_cordon" );
      if( spc != nullvalue_i ){
	vspgrapevineparameters.shoots_per_cordon = uint(spc);
      }

      float lsf = XMLloadfloat( s, "leaf_spacing_fraction" );
      if( lsf != nullvalue_f ){
	vspgrapevineparameters.leaf_spacing_fraction = lsf;
      }
      
      float gr = XMLloadfloat( s, "grape_radius" );
      if( gr != nullvalue_f ){
	vspgrapevineparameters.grape_radius = gr;
      }

      float clr = XMLloadfloat( s, "cluster_radius" );
      if( clr != nullvalue_f ){
	vspgrapevineparameters.cluster_radius = clr;
      }

      RGBcolor grape_color = XMLloadrgb( s, "grape_color");
      if( grape_color.r != nullvalue_f && grape_color.g != nullvalue_f && grape_color.b != nullvalue_f ){
	vspgrapevineparameters.grape_color = grape_color;
      }

      int grape_subdivisions = XMLloadint( s, "grape_subdivisions" );
      if( grape_subdivisions != nullvalue_i ){
	vspgrapevineparameters.grape_subdivisions = uint(grape_subdivisions);
      }

      float plant_spacing = XMLloadfloat( s, "plant_spacing" );
      if( plant_spacing != nullvalue_f ){
	vspgrapevineparameters.plant_spacing = plant_spacing;
      }

      float row_spacing = XMLloadfloat( s, "row_spacing" );
      if( row_spacing != nullvalue_f ){
	vspgrapevineparameters.row_spacing = row_spacing;
      }

      int2 plant_count = XMLloadint2( s, "");
      if( plant_count.x != nullvalue_f && plant_count.y != nullvalue_f ){
	vspgrapevineparameters.plant_count = plant_count;
      }

      vec3 canopy_origin = XMLloadvec3( s, "canopy_origin");
      if( canopy_origin.x != nullvalue_f && canopy_origin.y != nullvalue_f ){
	vspgrapevineparameters.canopy_origin = canopy_origin;
      }

      float canopy_rotation = XMLloadfloat( s, "canopy_rotation" );
      if( canopy_rotation != nullvalue_f ){
	vspgrapevineparameters.canopy_rotation = canopy_rotation;
      }


      buildCanopy( vspgrapevineparameters );
      
    }

    //Ground
    for (pugi::xml_node s = cgen.child("Ground"); s; s = s.next_sibling("Ground")){

      vec3 origin = XMLloadvec3( s, "origin");
      if( origin.x == nullvalue_f || origin.y == nullvalue_f || origin.z == nullvalue_f ){
	origin = make_vec3(0,0,0);
	if( printmessages ){
	  std::cout << "WARNING: origin not provided for ground in file " << filename << std::endl;
	}
      }

      vec2 extent = XMLloadvec2( s, "extent");
      if( extent.x == nullvalue_f || extent.y == nullvalue_f ){
	extent = make_vec2(1,1);
	if( printmessages ){
	  std::cout << "WARNING: horizontal extent not provided for ground in file " << filename << std::endl;
	}
      }

      int2 texture_subtiles = XMLloadint2( s, "texture_subtiles");
      if( texture_subtiles.x == nullvalue_i || texture_subtiles.y == nullvalue_i ){
	texture_subtiles = make_int2(1,1);
      }

      int2 texture_subpatches = XMLloadint2( s, "texture_subpatches");
      if( texture_subpatches.x == nullvalue_i || texture_subpatches.y == nullvalue_i ){
	texture_subpatches = make_int2(1,1);
      }

      std::string texturefile = XMLloadstring( s, "ground_texture_file");
      if( texturefile.compare(nullvalue_s)==0 ){
	texturefile = "plugins/canopygenerator/textures/dirt.jpg";
	if( printmessages ){
	  std::cout << "WARNING: texture map file not provided for ground in file " << filename << std::endl;
	}
      }

      float rotation = XMLloadfloat( s, "rotation");
      if( rotation == nullvalue_f ){
	rotation = 0;
      }

      buildGround( origin, extent, texture_subtiles, texture_subpatches, texturefile.c_str(), rotation );

    }
    
  }

  std::cout << "done." << std::endl;

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

  float Lmax = sqrtf(params.leaf_size.x*params.leaf_size.x + params.leaf_size.y*params.leaf_size.y);

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

    vec3 position;

      
    if(params.buffer == "z"){
      position = params.canopy_origin + make_vec3( (-0.5+rx)*params.canopy_extent.x, (-0.5+ry)*params.canopy_extent.y,0.5*Lmax+rz*(params.canopy_height-Lmax) );
    }else if(params.buffer == "xyz"){
      position = params.canopy_origin + make_vec3( 0.5*Lmax + (rx)*(params.canopy_extent.x-Lmax) + -0.5*params.canopy_extent.x,0.5*Lmax + (ry)*(params.canopy_extent.y-Lmax) + -0.5*params.canopy_extent.y,0.5*Lmax + rz*(params.canopy_height-Lmax) );        
    }else{
      position = params.canopy_origin + make_vec3( (-0.5+rx)*params.canopy_extent.x,(-0.5+ry)*params.canopy_extent.y,(rz)*(params.canopy_height) );
    }

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

  vec3 r = params.crown_radius;

  float solidFractionx;
  if(params.leaf_texture_file.size()==0){
    solidFractionx = 1.0;
  }else{
    helios::Texture texture(params.leaf_texture_file.c_str());
    solidFractionx = texture.getSolidFraction();
  }

  float leafArea = params.leaf_size.x*params.leaf_size.y*solidFractionx;
  int Nleaves = round(4.f/3.f*M_PI*r.x*r.y*r.z*params.leaf_area_density/leafArea);

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
	center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+(i+0.5)*params.plant_spacing.x, -0.5*canopy_extent.y+(j+0.5)*params.plant_spacing.y, r.z );
      }else if( cconfig.compare("random")==0 ){
	float rx = unif_distribution(generator);
	float ry = unif_distribution(generator);
	center = params.canopy_origin+make_vec3(-0.5*canopy_extent.x+i*params.plant_spacing.x+r.x+(params.plant_spacing.x-2.f*r.x)*rx, -0.5*canopy_extent.y+j*params.plant_spacing.y+r.y+(params.plant_spacing.y-2.f*r.y)*ry, r.z );
      }

      if( params.canopy_rotation!=0 ){
	center = rotatePointAboutLine( center, params.canopy_origin, make_vec3(0,0,1), params.canopy_rotation );
      }
	
      for (int l=0; l<Nleaves; l++ ){
	  
	// float u = unif_distribution(generator);
	// float v = unif_distribution(generator);
	// float theta = u * 2.0 * M_PI;
	// float phi = acosf(2.0 * v - 1.0);
	// float rad = pow(unif_distribution(generator),1.f/3.f);
	// float x = r * rad * sinf(phi) * cosf(theta);
	// float y = r * rad * sinf(phi) * sinf(theta);
	// float z = r * rad * cosf(phi);

	vec3 position(-9999,-9999,-9999);

	while( pow(position.x,2)/pow(params.crown_radius.x,2)+pow(position.y,2)/pow(params.crown_radius.y,2)+pow(position.z,2)/pow(params.crown_radius.z,2) > 1.f ){

	  float u = unif_distribution(generator);
	  float v = unif_distribution(generator);
	  float w = unif_distribution(generator);

	  position = make_vec3( (-1+2.f*u)*r.x, (-1+2.f*v)*r.y, (-1+2.f*w)*r.z );

	}

	float theta = sampleLeafPDF(params.leaf_angle_distribution.c_str());
	float phi = 2.f*M_PI*unif_distribution(generator);

	uint ID = context->copyObject(ID0);
	context->getObjectPointer(ID)->rotate(-theta,"y");
	context->getObjectPointer(ID)->rotate(phi,"z");
	context->getObjectPointer(ID)->translate(center+position);

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
