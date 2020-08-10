/** \file "grapevine.cpp" Definitions of functions for building grapevine plant geometries.
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

//!! To-Do Notes: 
//!! 1. Reduce default width of unilateral vines
//!! 2. Should add a plant width parameter to grapevines, because currently it just assumes same as plant spacing.

#include "CanopyGenerator.h"

using namespace helios;
using namespace std;

std::vector<std::vector<uint> > CanopyGenerator::addGrapeCluster( vec3 position, float grape_rad, float cluster_rad ){

  float z = position.z;

  float theta=0;
  float RB;
  int N = 100;
  int count=0;
  int max_grape_levels = 13-context->randu(0,4);

  vec3 xgrape;

  std::vector<std::vector<uint> > UUID;

  while( (N>3 || count<3) && count<max_grape_levels ){
    
      if(count<3){
	RB=cluster_rad*pow(0.5,2-count);
      }else if(count==3){
	RB=cluster_rad;
      }

      N=floor( M_PI*RB/grape_rad )+1;

      float dtheta=2.f*M_PI/N;
      for( int j=0; j<N; j++ ){
	
	xgrape.x = position.x+RB*sin(theta)+context->randn()*0.25*grape_rad;
	xgrape.y = position.y+RB*cos(theta)+context->randn()*0.25*grape_rad;
	xgrape.z = z+context->randn()*0.25*grape_rad;

	std::vector<uint> U=context->addSphere( 8, xgrape, grape_rad, make_RGBcolor(0.18,0.2,0.25) );
	UUID.push_back(U);
	
	theta=theta+dtheta;
	
      }
      z=z-1.8f*grape_rad;
      theta=theta+0.5*dtheta;
      RB=RB*0.85+context->randn()*RB*0.15;
      count=count+1;
  }

  return UUID;

}

void CanopyGenerator::grapevineVSP( const VSPGrapevineParameters params, const vec3 origin ){

  float mean_shoot_angle = 0.1*M_PI;

  vector<uint> U;
  std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
  std::vector<std::vector<uint> > UUID_leaf_plant;
  std::vector<std::vector<std::vector<uint> > > UUID_fruit_plant;

  std::uniform_real_distribution<float> unif_distribution;

  //------ trunks -------//

  std::vector<float> rad_main;
  rad_main.push_back(0.75*params.trunk_radius);
  rad_main.push_back(0.8f*params.trunk_radius);
  rad_main.push_back(1.f*params.trunk_radius);
  rad_main.push_back(0.7f*params.trunk_radius);
  rad_main.push_back(0.95f*params.trunk_radius);
  rad_main.push_back(0.1*params.trunk_radius);
  std::vector<vec3> pos_main;
  pos_main.push_back(make_vec3(0.,0.,0.0));
  pos_main.push_back(make_vec3(0,0,0.2f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.22f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.6f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.96f*params.trunk_height));
  pos_main.push_back(make_vec3(0.,0.,params.trunk_height));

  for( uint i=0; i<rad_main.size(); i++ ){
    pos_main.at(i) = pos_main.at(i) + origin;
  }
      
  UUID_trunk_plant = context->addTube(params.wood_subdivisions,pos_main,rad_main, params.wood_texture_file.c_str() );

  //---- Cordons -----//

  float diff = params.cordon_height-params.trunk_height;

  float cost = cosf(params.canopy_rotation);
  float sint = sinf(params.canopy_rotation);

  //West Cordon
  std::vector<float> rad_cordw;
  rad_cordw.push_back(params.cordon_radius);
  rad_cordw.push_back(0.95*params.cordon_radius);
  rad_cordw.push_back(0.9f*params.cordon_radius);
  rad_cordw.push_back(0.85f*params.cordon_radius);
  rad_cordw.push_back(0.8*params.cordon_radius);
  rad_cordw.push_back(0.6*params.cordon_radius);
  rad_cordw.push_back(0.2f*params.cordon_radius);
  std::vector<vec3> pos_cordw;
  pos_cordw.push_back(make_vec3(0.01f*0.5*params.plant_spacing*cost,0.01f*0.5*params.plant_spacing*sint,0.95*params.trunk_height));
  pos_cordw.push_back(make_vec3(0.05f*0.5*params.plant_spacing*cost,0.05f*0.5*params.plant_spacing*sint,params.trunk_height+0.1f*diff));
  pos_cordw.push_back(make_vec3(0.15f*0.5*params.plant_spacing*cost,0.15f*0.5*params.plant_spacing*sint,params.trunk_height+0.65f*diff));
  pos_cordw.push_back(make_vec3(0.45f*0.5*params.plant_spacing*cost,0.45f*0.5*params.plant_spacing*sint,params.trunk_height+0.95f*diff));
  pos_cordw.push_back(make_vec3(0.6f*0.5*params.plant_spacing*cost,0.6f*0.5*params.plant_spacing*sint,params.trunk_height+1.05f*diff));
  pos_cordw.push_back(make_vec3(0.85f*0.5*params.plant_spacing*cost,0.85f*0.5*params.plant_spacing*sint,params.trunk_height+diff));
  pos_cordw.push_back(make_vec3(1.0f*0.5*params.plant_spacing*cost,1.0f*0.5*params.plant_spacing*sint,params.trunk_height+diff));

  std::vector<vec3> tmp;
  tmp.resize(pos_cordw.size());
  for( uint i=0; i<pos_cordw.size(); i++ ){
    tmp.at(i) = pos_cordw.at(i) + origin;
  }

  UUID_branch_plant = context->addTube(params.wood_subdivisions,tmp,rad_cordw,params.wood_texture_file.c_str() );

  //East Cordon
  std::vector<float> rad_corde;
  rad_corde.push_back(params.cordon_radius);
  rad_corde.push_back(0.95*params.cordon_radius);
  rad_corde.push_back(0.9f*params.cordon_radius);
  rad_corde.push_back(0.85f*params.cordon_radius);
  rad_corde.push_back(0.8*params.cordon_radius);
  rad_corde.push_back(0.6*params.cordon_radius);
  rad_corde.push_back(0.2f*params.cordon_radius);
  std::vector<vec3> pos_corde;
  pos_corde.push_back(make_vec3(-0.01f*0.5*params.plant_spacing*cost,-0.01f*0.5*params.plant_spacing*sint,0.95*params.trunk_height));
  pos_corde.push_back(make_vec3(-0.05f*0.5*params.plant_spacing*cost,-0.05f*0.5*params.plant_spacing*sint,params.trunk_height+0.1f*diff));
  pos_corde.push_back(make_vec3(-0.15f*0.5*params.plant_spacing*cost,-0.15f*0.5*params.plant_spacing*sint,params.trunk_height+0.65f*diff));
  pos_corde.push_back(make_vec3(-0.45f*0.5*params.plant_spacing*cost,-0.45f*0.5*params.plant_spacing*sint,params.trunk_height+0.95f*diff));
  pos_corde.push_back(make_vec3(-0.6f*0.5*params.plant_spacing*cost,-0.6f*0.5*params.plant_spacing*sint,params.trunk_height+1.05f*diff));
  pos_corde.push_back(make_vec3(-0.85f*0.5*params.plant_spacing*cost,-0.85f*0.5*params.plant_spacing*sint,params.trunk_height+diff));
  pos_corde.push_back(make_vec3(-1.0f*0.5*params.plant_spacing*cost,-1.0f*0.5*params.plant_spacing*sint,params.trunk_height+diff));

  tmp.resize(pos_corde.size());
  for( uint i=0; i<pos_corde.size(); i++ ){
    tmp.at(i) = pos_corde.at(i) + origin;
  }

  U = context->addTube(params.wood_subdivisions,tmp,rad_corde,params.wood_texture_file.c_str() );
  UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end() );
		      
  //------- primary shoots ---------//

  float height = params.cordon_height + params.shoot_length;
  
  for( uint c=0; c<2; c++ ){

    std::vector<float> rad_cord;
    std::vector<vec3> pos_cord;
    float sign;
    if( c==0 ){
      pos_cord = pos_cordw;
      rad_cord = rad_cordw;
      sign = 1;
    }else{
      pos_cord = pos_corde;
      rad_cord = rad_corde;
      sign = -1;
    }

    float dx = fabs(pos_cord.back().y-pos_cord.at(0).y)/(float(params.shoots_per_cordon));

    for( int j=1; j<params.shoots_per_cordon+1; j++ ){
	
      float frac_shoot = float(j)/float(params.shoots_per_cordon);

      vec3 cane_base = interpolateTube( pos_cord, frac_shoot );

      std::vector<float> rad_pshoot;
      std::vector<vec3> pos_pshoot;
	  
      //cane base
      rad_pshoot.push_back( params.shoot_radius );
      pos_pshoot.push_back( cane_base );
      
      //cane nodes
      float phirot = (0.5f-unif_distribution(generator))*0.5*M_PI;
      phirot=0.5*M_PI;
      if( unif_distribution(generator)<0.5 ){
  	phirot+=M_PI;
      }

      float theta0 = (0.3f-unif_distribution(generator))*0.6;
      float theta_end = (0.1f-unif_distribution(generator))*0.2;
	  
      uint Nz = 2*params.wood_subdivisions;
      float dz = ((1+getVariation(0.1,generator))*height-params.cordon_height)/float(Nz);
      for( uint k=1; k<Nz; k++ ){
	    
  	vec3 n = rotatePoint( make_vec3(0,0,dz), mean_shoot_angle*M_PI/180.f*(theta0+1.2*float(k)/float(Nz-1)), phirot );

  	pos_pshoot.push_back( pos_pshoot.back()+n+make_vec3(getVariation(0.02,generator),getVariation(0.01,generator),0) );
	
  	rad_pshoot.push_back(params.shoot_radius);
	
      }

      rad_pshoot.back()=0.01f;

      std::vector<vec3> tmp;
      tmp.resize(pos_pshoot.size());
      for( uint i=0; i<pos_pshoot.size(); i++ ){
  	tmp.at(i) = pos_pshoot.at(i) + origin;
      }
	  
      U = context->addTube(params.wood_subdivisions,tmp,rad_pshoot, params.wood_texture_file.c_str() );
      UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end() );

      //grape clusters
      std::vector<std::vector<uint> > UUID_grapes;
      if( params.grape_radius>0 && params.cluster_radius>0 ){
      
	float fgrape = fmax(0.1,0.2+getVariation(0.2,generator));
	vec3 p_grape = interpolateTube( tmp, fgrape );
	int sgn=1;
	if( unif_distribution(generator)<0.5 ){
	  sgn = -1;
	}
	vec3 offset(sgn*getVariation(0.1,generator)*sint,sgn*getVariation(0.1,generator)*cost,0.f);
	UUID_grapes = addGrapeCluster( p_grape+offset, params.grape_radius, params.cluster_radius );
      }
      UUID_fruit_plant.push_back( UUID_grapes );

      //leaves
      float flip = 0;
      if( unif_distribution(generator)<0.5 ){
	flip = 1;
      }
      float lfrac = 1.f;
      while( lfrac>0.*params.leaf_width ){

  	float lsize = fmaxf(params.leaf_width*(1.f-exp(-5.f*(1-lfrac))),0.1*params.leaf_width);
	
  	vec3 pos_leaf = interpolateTube( pos_pshoot, lfrac );

  	vec3 parent_normal = interpolateTube( pos_pshoot, fmax(0,lfrac-0.001) )-pos_leaf;
  	parent_normal.normalize();
  	vec3 leaf_offset = rotatePointAboutLine(make_vec3(0,lsize*(0.3+getVariation(0.25,generator)),0), make_vec3(0,0,0), parent_normal, flip*M_PI+unif_distribution(generator)*0.25*M_PI );

  	float s;
  	if( int(flip)%2==0 ){
  	  s = 1;
  	}else{
  	  s = -1;
  	}

  	float Rphi = -params.canopy_rotation - s*0.5*M_PI*(1.f+getVariation(0.4,generator));
  	float Rtheta = 0.4*M_PI*(1.f+getVariation(0.1,generator));

  	vec3 position = origin+pos_leaf+leaf_offset;

  	UUID_leaf_plant.push_back( context->addTile( make_vec3(0,0,0), make_vec2(lsize,lsize), make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() ) );
	
  	context->rotatePrimitive(UUID_leaf_plant.back(),-Rtheta,"y");
  	context->rotatePrimitive(UUID_leaf_plant.back(),Rphi,"z");
  	context->translatePrimitive(UUID_leaf_plant.back(),position);

	lfrac = lfrac - params.leaf_spacing_fraction*lsize*(1.f+getVariation(0.25,generator));
	
  	flip++;
			
      }
	    
    }
	
  }

  UUID_trunk.push_back( UUID_trunk_plant );
  UUID_branch.push_back( UUID_branch_plant );
  UUID_leaf.push_back( UUID_leaf_plant );
  UUID_fruit.push_back( UUID_fruit_plant );

}

void CanopyGenerator::grapevineSplit( const SplitGrapevineParameters params, const vec3 origin ){

  vector<uint> U;
  std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
  std::vector<std::vector<uint> > UUID_leaf_plant;
  std::vector<std::vector<std::vector<uint> > > UUID_fruit_plant;

  std::uniform_real_distribution<float> unif_distribution;

  //------ trunks -------//

  std::vector<float> rad_main;
  rad_main.push_back(0.75*params.trunk_radius);
  rad_main.push_back(0.8f*params.trunk_radius);
  rad_main.push_back(1.f*params.trunk_radius);
  rad_main.push_back(0.7f*params.trunk_radius);
  rad_main.push_back(0.95f*params.trunk_radius);
  rad_main.push_back(0.1*params.trunk_radius);
  std::vector<vec3> pos_main;
  pos_main.push_back(make_vec3(0.,0.,0.0));
  pos_main.push_back(make_vec3(0,0,0.2f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.22f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.6f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.96f*params.trunk_height));
  pos_main.push_back(make_vec3(0.,0.,params.trunk_height));

  for( uint i=0; i<rad_main.size(); i++ ){
    pos_main.at(i) = pos_main.at(i) + origin;
  }
      
  UUID_trunk_plant = context->addTube(params.wood_subdivisions,pos_main,rad_main, params.wood_texture_file.c_str() );

  //------ crown -------//

  float diff = params.cordon_height-params.trunk_height;

  float cost = cosf(params.canopy_rotation+0.5*M_PI);
  float sint = sinf(params.canopy_rotation+0.5*M_PI);

  std::vector<float> rad_crown;
  rad_crown.push_back(0.6*params.trunk_radius);
  rad_crown.push_back(0.55f*params.trunk_radius);
  rad_crown.push_back(0.5f*params.trunk_radius);
  rad_crown.push_back(0.45f*params.trunk_radius);
  rad_crown.push_back(0.4f*params.trunk_radius);
  
  std::vector<vec3> pos_crownw;
  pos_crownw.push_back(make_vec3(0.,0.,0.95*params.trunk_height));
  pos_crownw.push_back(make_vec3(0.05f*0.5*params.cordon_spacing*cost,0.05f*0.5*params.cordon_spacing*sint,params.trunk_height));
  pos_crownw.push_back(make_vec3(0.25f*0.5*params.cordon_spacing*cost,0.25f*0.5*params.cordon_spacing*sint,params.trunk_height+0.1f*diff));
  pos_crownw.push_back(make_vec3(0.45f*0.5*params.cordon_spacing*cost,0.45f*0.5*params.cordon_spacing*sint,params.trunk_height+0.65f*diff));
  pos_crownw.push_back(make_vec3(0.75f*0.5*params.cordon_spacing*cost,0.75f*0.5*params.cordon_spacing*sint,params.cordon_height));

  for( uint i=0; i<rad_crown.size(); i++ ){
    pos_crownw.at(i) = pos_crownw.at(i) + origin;
  }
      
  UUID_trunk_plant = context->addTube(params.wood_subdivisions,pos_crownw,rad_crown, params.wood_texture_file.c_str() );

  std::vector<vec3> pos_crowne;
  pos_crowne.push_back(make_vec3(0.,0.,0.95*params.trunk_height));
  pos_crowne.push_back(make_vec3(-0.05f*0.5*params.cordon_spacing*cost,-0.05f*0.5*params.cordon_spacing*sint,params.trunk_height));
  pos_crowne.push_back(make_vec3(-0.25f*0.5*params.cordon_spacing*cost,-0.25f*0.5*params.cordon_spacing*sint,params.trunk_height+0.1f*diff));
  pos_crowne.push_back(make_vec3(-0.45f*0.5*params.cordon_spacing*cost,-0.45f*0.5*params.cordon_spacing*sint,params.trunk_height+0.65f*diff));
  pos_crowne.push_back(make_vec3(-0.75f*0.5*params.cordon_spacing*cost,-0.75f*0.5*params.cordon_spacing*sint,params.cordon_height));

  for( uint i=0; i<rad_crown.size(); i++ ){
    pos_crowne.at(i) = pos_crowne.at(i) + origin;
  }
      
  UUID_trunk_plant = context->addTube(params.wood_subdivisions,pos_crowne,rad_crown, params.wood_texture_file.c_str() );

  //---- Cordons -----//

  std::vector<float> rad_cord;
  rad_cord.push_back(params.cordon_radius);
  rad_cord.push_back(0.95*params.cordon_radius);
  rad_cord.push_back(0.9f*params.cordon_radius);
  rad_cord.push_back(0.9f*params.cordon_radius);
  rad_cord.push_back(0.9*params.cordon_radius);
  rad_cord.push_back(0.6*params.cordon_radius);
  rad_cord.push_back(0.2f*params.cordon_radius);
  
  //West Cordon
  std::vector<vec3> pos_cordnw;
  pos_cordnw.push_back(make_vec3(0.7f*0.5*params.cordon_spacing*cost,0.7f*0.5*params.cordon_spacing*sint,0.99*params.cordon_height));
  pos_cordnw.push_back(make_vec3(0.85f*0.5*params.cordon_spacing*cost+0.025*sint,0.85f*0.5*params.cordon_spacing*sint+0.025*cost,params.cordon_height));
  pos_cordnw.push_back(make_vec3(0.95f*0.5*params.cordon_spacing*cost+0.075*sint,0.95f*0.5*params.cordon_spacing*sint+0.075*cost,params.cordon_height));
  pos_cordnw.push_back(make_vec3(0.5*params.cordon_spacing*cost+0.12*sint,0.5*params.cordon_spacing*sint+0.12*cost,params.cordon_height));
  pos_cordnw.push_back(make_vec3(0.5*params.cordon_spacing*cost+0.4*0.5*params.plant_spacing*sint,0.5*params.cordon_spacing*sint+0.4*0.5*params.plant_spacing*cost,0.94*params.cordon_height));
  pos_cordnw.push_back(make_vec3(0.5*params.cordon_spacing*cost+0.8*0.5*params.plant_spacing*sint,0.5*params.cordon_spacing*sint+0.8*0.5*params.plant_spacing*cost,0.97*params.cordon_height));
  pos_cordnw.push_back(make_vec3(0.5*params.cordon_spacing*cost+0.5*params.plant_spacing*sint,0.5*params.cordon_spacing*sint+0.5*params.plant_spacing*cost,params.cordon_height));

  std::vector<vec3> tmp;
  tmp.resize(pos_cordnw.size());
  for( uint i=0; i<pos_cordnw.size(); i++ ){
    tmp.at(i) = pos_cordnw.at(i) + origin;
  }

  UUID_branch_plant = context->addTube(params.wood_subdivisions,tmp,rad_cord,params.wood_texture_file.c_str() );

  std::vector<vec3> pos_cordsw;
  pos_cordsw.push_back(make_vec3(0.7f*0.5*params.cordon_spacing*cost,0.7f*0.5*params.cordon_spacing*sint,0.99*params.cordon_height));
  pos_cordsw.push_back(make_vec3(0.85f*0.5*params.cordon_spacing*cost-0.025*sint,0.85f*0.5*params.cordon_spacing*sint-0.025*cost,params.cordon_height));
  pos_cordsw.push_back(make_vec3(0.95f*0.5*params.cordon_spacing*cost-0.075*sint,0.95f*0.5*params.cordon_spacing*sint-0.075*cost,params.cordon_height));
  pos_cordsw.push_back(make_vec3(0.5*params.cordon_spacing*cost-0.12*sint,0.5*params.cordon_spacing*sint-0.12*cost,params.cordon_height));
  pos_cordsw.push_back(make_vec3(0.5*params.cordon_spacing*cost-0.4*0.5*params.plant_spacing*sint,0.5*params.cordon_spacing*sint-0.4*0.5*params.plant_spacing*cost,0.94*params.cordon_height));
  pos_cordsw.push_back(make_vec3(0.5*params.cordon_spacing*cost-0.8*0.5*params.plant_spacing*sint,0.5*params.cordon_spacing*sint-0.8*0.5*params.plant_spacing*cost,0.97*params.cordon_height));
  pos_cordsw.push_back(make_vec3(0.5*params.cordon_spacing*cost-0.5*params.plant_spacing*sint,0.5*params.cordon_spacing*sint-0.5*params.plant_spacing*cost,params.cordon_height));

  tmp.resize(pos_cordsw.size());
  for( uint i=0; i<pos_cordsw.size(); i++ ){
    tmp.at(i) = pos_cordsw.at(i) + origin;
  }

  U = context->addTube(params.wood_subdivisions,tmp,rad_cord,params.wood_texture_file.c_str() );
  UUID_branch_plant.insert( UUID_branch_plant.end(), U.begin(), U.end() );

  //East Cordon
  std::vector<vec3> pos_cordne;
  pos_cordne.push_back(make_vec3(-0.7f*0.5*params.cordon_spacing*cost,-0.7f*0.5*params.cordon_spacing*sint,0.99*params.cordon_height));
  pos_cordne.push_back(make_vec3(-0.85f*0.5*params.cordon_spacing*cost+0.025*sint,-0.85f*0.5*params.cordon_spacing*sint+0.025*cost,params.cordon_height));
  pos_cordne.push_back(make_vec3(-0.95f*0.5*params.cordon_spacing*cost+0.075*sint,-0.95f*0.5*params.cordon_spacing*sint+0.075*cost,params.cordon_height));
  pos_cordne.push_back(make_vec3(-0.5*params.cordon_spacing*cost+0.12*sint,-0.5*params.cordon_spacing*sint+0.12*cost,params.cordon_height));
  pos_cordne.push_back(make_vec3(-0.5*params.cordon_spacing*cost+0.4*0.5*params.plant_spacing*sint,-0.5*params.cordon_spacing*sint+0.4*0.5*params.plant_spacing*cost,0.94*params.cordon_height));
  pos_cordne.push_back(make_vec3(-0.5*params.cordon_spacing*cost+0.8*0.5*params.plant_spacing*sint,-0.5*params.cordon_spacing*sint+0.8*0.5*params.plant_spacing*cost,0.97*params.cordon_height));
  pos_cordne.push_back(make_vec3(-0.5*params.cordon_spacing*cost+0.5*params.plant_spacing*sint,-0.5*params.cordon_spacing*sint+0.5*params.plant_spacing*cost,params.cordon_height));

  tmp.resize(pos_cordne.size());
  for( uint i=0; i<pos_cordne.size(); i++ ){
    tmp.at(i) = pos_cordne.at(i) + origin;
  }

  U = context->addTube(params.wood_subdivisions,tmp,rad_cord,params.wood_texture_file.c_str() );
  UUID_branch_plant.insert( UUID_branch_plant.end(), U.begin(), U.end() );

  std::vector<vec3> pos_cordse;
  pos_cordse.push_back(make_vec3(-0.7f*0.5*params.cordon_spacing*cost,-0.7f*0.5*params.cordon_spacing*sint,0.99*params.cordon_height));
  pos_cordse.push_back(make_vec3(-0.85f*0.5*params.cordon_spacing*cost-0.025*sint,-0.85f*0.5*params.cordon_spacing*sint-0.025*cost,params.cordon_height));
  pos_cordse.push_back(make_vec3(-0.95f*0.5*params.cordon_spacing*cost-0.075*sint,-0.95f*0.5*params.cordon_spacing*sint-0.075*cost,params.cordon_height));
  pos_cordse.push_back(make_vec3(-0.5*params.cordon_spacing*cost-0.12*sint,-0.5*params.cordon_spacing*sint-0.12*cost,params.cordon_height));
  pos_cordse.push_back(make_vec3(-0.5*params.cordon_spacing*cost-0.4*0.5*params.plant_spacing*sint,-0.5*params.cordon_spacing*sint-0.4*0.5*params.plant_spacing*cost,0.94*params.cordon_height));
  pos_cordse.push_back(make_vec3(0.5*params.cordon_spacing*cost-0.8*0.5*params.plant_spacing*sint,-0.5*params.cordon_spacing*sint-0.8*0.5*params.plant_spacing*cost,0.97*params.cordon_height));
  pos_cordse.push_back(make_vec3(0.5*params.cordon_spacing*cost-0.5*params.plant_spacing*sint,-0.5*params.cordon_spacing*sint-0.5*params.plant_spacing*cost,params.cordon_height));

  tmp.resize(pos_cordse.size());
  for( uint i=0; i<pos_cordse.size(); i++ ){
    tmp.at(i) = pos_cordse.at(i) + origin;
  }

  U = context->addTube(params.wood_subdivisions,tmp,rad_cord,params.wood_texture_file.c_str() );
  UUID_branch_plant.insert( UUID_branch_plant.end(), U.begin(), U.end() );

  //------- primary shoots ---------//

  float height = params.cordon_height + params.shoot_length;

  for( uint d=0; d<2; d++ ){
  for( uint c=0; c<2; c++ ){

    std::vector<vec3> pos_cord;
    float sign;
    if( c==0 ){
      if( d==0 ){
	pos_cord = pos_cordsw;
      }else{
	pos_cord = pos_cordnw;
      }
      sign = 1;
    }else{
      if( d==0 ){
	pos_cord = pos_cordse;
      }else{
	pos_cord = pos_cordne;
      }
      sign = -1;
    }

    float dx = fabs(pos_cord.back().y-pos_cord.at(0).y)/(float(params.shoots_per_cordon));

    for( int j=1; j<params.shoots_per_cordon+1; j++ ){
	
      float frac_shoot = float(j)/float(params.shoots_per_cordon);

      vec3 cane_base = interpolateTube( pos_cord, frac_shoot );

      std::vector<float> rad_pshoot;
      std::vector<vec3> pos_pshoot;
	  
      //cane base
      rad_pshoot.push_back( params.shoot_radius );
      pos_pshoot.push_back( cane_base );
      
      //cane nodes
      float phirot=0.5*M_PI*(1+(-0.5+unif_distribution(generator))*1.0);
      if( unif_distribution(generator)<0.5 ){
  	phirot+=M_PI;
      }

      float theta0 = params.shoot_angle_base*(1.f+(-0.5+unif_distribution(generator))*0.6);
      float theta_end = params.shoot_angle_tip*(1.f+(-0.5+unif_distribution(generator))*0.6);
	  
      uint Nz = 2*params.wood_subdivisions;
      float dz = ((1+getVariation(0.1,generator))*height-params.cordon_height)/float(Nz);
      for( uint k=1; k<Nz; k++ ){
	    
  	vec3 n = rotatePoint( make_vec3(0,0,dz), (theta0+(theta_end-theta0)*float(k)/float(Nz-1)), phirot );

  	pos_pshoot.push_back( pos_pshoot.back()+n+make_vec3(getVariation(0.02,generator),getVariation(0.01,generator),0) );
	
  	rad_pshoot.push_back(params.shoot_radius);
	
      }

      rad_pshoot.back()=0.01f;

      std::vector<vec3> tmp;
      tmp.resize(pos_pshoot.size());
      for( uint i=0; i<pos_pshoot.size(); i++ ){
  	tmp.at(i) = pos_pshoot.at(i) + origin;
      }
	  
      U = context->addTube(params.wood_subdivisions,tmp,rad_pshoot, params.wood_texture_file.c_str() );
      UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end() );

      //grape clusters
      std::vector<std::vector<uint> > UUID_grapes;
      if( params.grape_radius>0 && params.cluster_radius>0 ){
      
	float fgrape = fmax(0.1,0.2+getVariation(0.2,generator));
	vec3 p_grape = interpolateTube( tmp, fgrape );
	int sgn=1;
	if( unif_distribution(generator)<0.5 ){
	  sgn = -1;
	}
	vec3 offset(sgn*getVariation(0.1,generator)*sint,sgn*getVariation(0.1,generator)*cost,0.f);
	UUID_grapes = addGrapeCluster( p_grape+offset, params.grape_radius, params.cluster_radius );
      }
      UUID_fruit_plant.push_back( UUID_grapes );
	
      //leaves
      float flip = 0;
      if( unif_distribution(generator)<0.5 ){
	flip = 1;
      }
      float lfrac = 1.f;
      while( lfrac>0.*params.leaf_width ){

  	float lsize = fmaxf(params.leaf_width*(1.f-exp(-5.f*(1-lfrac))),0.1*params.leaf_width);
	
  	vec3 pos_leaf = interpolateTube( pos_pshoot, lfrac );

  	vec3 parent_normal = interpolateTube( pos_pshoot, fmax(0,lfrac-0.001) )-pos_leaf;
  	parent_normal.normalize();
  	vec3 leaf_offset = rotatePointAboutLine(make_vec3(0,lsize*(0.3+getVariation(0.25,generator)),0), make_vec3(0,0,0), parent_normal, flip*M_PI+unif_distribution(generator)*0.25*M_PI );

  	float s;
  	if( int(flip)%2==0 ){
  	  s = 1;
  	}else{
  	  s = -1;
  	}

  	float Rphi = -params.canopy_rotation - s*0.5*M_PI*(1.f+getVariation(0.4,generator));
  	float Rtheta = 0.4*M_PI*(1.f+getVariation(0.1,generator));

  	vec3 position = origin+pos_leaf+leaf_offset;

  	UUID_leaf_plant.push_back( context->addTile( make_vec3(0,0,0), make_vec2(lsize,lsize), make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() ) );
	
  	context->rotatePrimitive(UUID_leaf_plant.back(),-Rtheta,"y");
  	context->rotatePrimitive(UUID_leaf_plant.back(),Rphi,"z");
  	context->translatePrimitive(UUID_leaf_plant.back(),position);

	lfrac = lfrac - params.leaf_spacing_fraction*lsize*(1.f+getVariation(0.25,generator));
	
  	flip++;
			
      }
	    
    }
	
  }
  }

  UUID_trunk.push_back( UUID_trunk_plant );
  UUID_branch.push_back( UUID_branch_plant );
  UUID_leaf.push_back( UUID_leaf_plant );
  UUID_fruit.push_back( UUID_fruit_plant );

}

void CanopyGenerator::grapevineUnilateral( const UnilateralGrapevineParameters params, const vec3 origin){

  float mean_shoot_angle = 0.1*M_PI;

  vector<uint> U;
  std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
  std::vector<std::vector<uint> > UUID_leaf_plant;
  std::vector<std::vector<std::vector<uint> > > UUID_fruit_plant;

  std::uniform_real_distribution<float> unif_distribution;

  float cost = cosf(params.canopy_rotation);
  float sint = sinf(params.canopy_rotation);
  
  //------ trunks -------//

  std::vector<float> rad_main;
  rad_main.push_back(0.75*params.trunk_radius);
  rad_main.push_back(0.8f*params.trunk_radius);
  rad_main.push_back(1.f*params.trunk_radius);
  rad_main.push_back(0.7f*params.trunk_radius);
  rad_main.push_back(0.95f*params.trunk_radius);
  rad_main.push_back(0.1*params.trunk_radius);
  std::vector<vec3> pos_main;
  pos_main.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius)*sint,0.0));
  pos_main.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius)*sint,0.2f*params.trunk_height));
  pos_main.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius)*sint,0.22f*params.trunk_height));
  pos_main.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius)*sint,0.6f*params.trunk_height));
  pos_main.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius)*sint,0.96f*params.trunk_height));
  pos_main.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius)*sint,params.trunk_height));

  for( uint i=0; i<rad_main.size(); i++ ){
    pos_main.at(i) = pos_main.at(i) + origin;
  }
      
  UUID_trunk_plant = context->addTube(params.wood_subdivisions,pos_main,rad_main, params.wood_texture_file.c_str() );

  //---- Cordons -----//

  float diff = params.cordon_height-params.trunk_height;

  //Cordon
  std::vector<float> rad_cord;
  rad_cord.push_back(params.cordon_radius);
  rad_cord.push_back(0.95*params.cordon_radius);
  rad_cord.push_back(0.9f*params.cordon_radius);
  rad_cord.push_back(0.85f*params.cordon_radius);
  rad_cord.push_back(0.8*params.cordon_radius);
  rad_cord.push_back(0.6*params.cordon_radius);
  rad_cord.push_back(0.2f*params.cordon_radius);
  std::vector<vec3> pos_cord;
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+0.01f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+0.01f*params.plant_spacing)*sint,0.95*params.trunk_height));
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+0.05f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+0.05f*params.plant_spacing)*sint,params.trunk_height+0.1f*diff));
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+0.15f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+0.15f*params.plant_spacing)*sint,params.trunk_height+0.65f*diff));
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+0.45f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+0.45f*params.plant_spacing)*sint,params.trunk_height+0.95f*diff));
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+0.6f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+0.6f*params.plant_spacing)*sint,params.trunk_height+1.05f*diff));
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+0.85f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+0.85f*params.plant_spacing)*sint,params.trunk_height+diff));
  pos_cord.push_back(make_vec3((-0.5*params.plant_spacing+0.5*params.trunk_radius+1.0f*params.plant_spacing)*cost,(-0.5*params.plant_spacing+0.5*params.trunk_radius+1.0f*params.plant_spacing)*sint,params.trunk_height+diff));

  std::vector<vec3> tmp;
  tmp.resize(pos_cord.size());
  for( uint i=0; i<pos_cord.size(); i++ ){
    tmp.at(i) = pos_cord.at(i) + origin;
  }

  UUID_branch_plant = context->addTube(params.wood_subdivisions,tmp,rad_cord,params.wood_texture_file.c_str() );
		      
  //------- primary shoots ---------//

  float height = params.cordon_height + params.shoot_length;
  
  float dx = fabs(pos_cord.back().y-pos_cord.at(0).y)/(float(params.shoots_per_cordon));

  for( int j=1; j<params.shoots_per_cordon+1; j++ ){
	
    float frac_shoot = float(j)/float(params.shoots_per_cordon);

    vec3 cane_base = interpolateTube( pos_cord, frac_shoot );
    
    std::vector<float> rad_pshoot;
    std::vector<vec3> pos_pshoot;
    
    //cane base
    rad_pshoot.push_back( params.shoot_radius );
    pos_pshoot.push_back( cane_base );
    
    //cane nodes
    float phirot = (0.5f-unif_distribution(generator))*0.5*M_PI;
    phirot=0.5*M_PI;
    if( unif_distribution(generator)<0.5 ){
      phirot+=M_PI;
    }
    
    float theta0 = (0.3f-unif_distribution(generator))*0.6;
    float theta_end = (0.1f-unif_distribution(generator))*0.2;
	  
    uint Nz = 2*params.wood_subdivisions;
    float dz = ((1+getVariation(0.1,generator))*height-params.cordon_height)/float(Nz);
    for( uint k=1; k<Nz; k++ ){
      
      vec3 n = rotatePoint( make_vec3(0,0,dz), mean_shoot_angle*M_PI/180.f*(theta0+1.2*float(k)/float(Nz-1)), phirot );

      pos_pshoot.push_back( pos_pshoot.back()+n+make_vec3(getVariation(0.02,generator),getVariation(0.01,generator),0) );
	
      rad_pshoot.push_back(params.shoot_radius);
	
    }

    rad_pshoot.back()=0.01f;

    std::vector<vec3> tmp;
    tmp.resize(pos_pshoot.size());
    for( uint i=0; i<pos_pshoot.size(); i++ ){
      tmp.at(i) = pos_pshoot.at(i) + origin;
    }
	  
    U = context->addTube(params.wood_subdivisions,tmp,rad_pshoot, params.wood_texture_file.c_str() );
    UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end() );

    //grape clusters
    std::vector<std::vector<uint> > UUID_grapes;
    if( params.grape_radius>0 && params.cluster_radius>0 ){
      
      float fgrape = fmax(0.1,0.2+getVariation(0.2,generator));
      vec3 p_grape = interpolateTube( tmp, fgrape );
      int sgn=1;
      if( unif_distribution(generator)<0.5 ){
	sgn = -1;
      }
      vec3 offset(sgn*getVariation(0.1,generator)*sint,sgn*getVariation(0.1,generator)*cost,0.f);
      UUID_grapes = addGrapeCluster( p_grape+offset, params.grape_radius, params.cluster_radius );
    }
    UUID_fruit_plant.push_back( UUID_grapes );
    
    //leaves
    float flip = 0;
    if( unif_distribution(generator)<0.5 ){
      flip = 1;
    }
    float lfrac = 1.f;
    while( lfrac>0.*params.leaf_width ){
      
      float lsize = fmaxf(params.leaf_width*(1.f-exp(-5.f*(1-lfrac))),0.1*params.leaf_width);
	
      vec3 pos_leaf = interpolateTube( pos_pshoot, lfrac );

      vec3 parent_normal = interpolateTube( pos_pshoot, fmax(0,lfrac-0.001) )-pos_leaf;
      parent_normal.normalize();
      vec3 leaf_offset = rotatePointAboutLine(make_vec3(0,lsize*(0.3+getVariation(0.25,generator)),0), make_vec3(0,0,0), parent_normal, flip*M_PI+unif_distribution(generator)*0.25*M_PI );
      
      float s;
      if( int(flip)%2==0 ){
	s = 1;
      }else{
	s = -1;
      }

      float Rphi = -params.canopy_rotation - s*0.5*M_PI*(1.f+getVariation(0.4,generator));
      float Rtheta = 0.4*M_PI*(1.f+getVariation(0.1,generator));
      
      vec3 position = origin+pos_leaf+leaf_offset;
      
      UUID_leaf_plant.push_back( context->addTile( make_vec3(0,0,0), make_vec2(lsize,lsize), make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() ) );
      
      context->rotatePrimitive(UUID_leaf_plant.back(),-Rtheta,"y");
      context->rotatePrimitive(UUID_leaf_plant.back(),Rphi,"z");
      context->translatePrimitive(UUID_leaf_plant.back(),position);
      
      lfrac = lfrac - params.leaf_spacing_fraction*lsize*(1.f+getVariation(0.25,generator));
      
      flip++;
				    
    }
	
  }

  UUID_trunk.push_back( UUID_trunk_plant );
  UUID_branch.push_back( UUID_branch_plant );
  UUID_leaf.push_back( UUID_leaf_plant );
  UUID_fruit.push_back( UUID_fruit_plant );

}

void CanopyGenerator::grapevineGoblet( const GobletGrapevineParameters params, const vec3 origin ){

  vector<uint> U;
  std::vector<uint> UUID_trunk_plant, UUID_branch_plant;
  std::vector<std::vector<uint> > UUID_leaf_plant;
  std::vector<std::vector<std::vector<uint> > > UUID_fruit_plant;

  std::uniform_real_distribution<float> unif_distribution;

  //------ trunks -------//

  std::vector<float> rad_main;
  rad_main.push_back(0.75*params.trunk_radius);
  rad_main.push_back(0.8f*params.trunk_radius);
  rad_main.push_back(1.f*params.trunk_radius);
  rad_main.push_back(0.7f*params.trunk_radius);
  rad_main.push_back(0.95f*params.trunk_radius);
  rad_main.push_back(0.1*params.trunk_radius);
  std::vector<vec3> pos_main;
  pos_main.push_back(make_vec3(0.,0.,0.0));
  pos_main.push_back(make_vec3(0,0,0.2f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.22f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.6f*params.trunk_height));
  pos_main.push_back(make_vec3(0,0,0.96f*params.trunk_height));
  pos_main.push_back(make_vec3(0.,0.,params.trunk_height));

  for( uint i=0; i<rad_main.size(); i++ ){
    pos_main.at(i) = pos_main.at(i) + origin;
  }
      
  UUID_trunk_plant = context->addTube(params.wood_subdivisions,pos_main,rad_main, params.wood_texture_file.c_str() );
		      
  //------- primary shoots ---------//
  
  for( uint c=0; c<2; c++ ){

    for( int j=1; j<params.shoots_per_cordon+1; j++ ){
	
      float frac_shoot = float(j)/float(params.shoots_per_cordon);

      float height = (params.cordon_height + params.shoot_length)*(1.f+0.55*(1-frac_shoot));

      vec3 cane_base = make_vec3(0.,0.,params.trunk_height);

      std::vector<float> rad_pshoot;
      std::vector<vec3> pos_pshoot;
	  
      //cane base
      rad_pshoot.push_back( params.shoot_radius );
      pos_pshoot.push_back( cane_base );

      float theta0 = 0.5*M_PI*(1.f-float(j-1)/float(params.shoots_per_cordon));
      float theta_end = (0.1f-unif_distribution(generator))*0.2;
	  
      uint Nz = 2*params.wood_subdivisions;
      float dz = ((1+getVariation(0.1,generator))*height-params.cordon_height)/float(Nz);
      for( uint k=1; k<Nz; k++ ){

	vec3 n = rotatePoint( make_vec3(0,0,dz), (theta0+(theta_end-theta0)*float(k)/float(Nz-1)), params.canopy_rotation+M_PI*float(c) );

  	pos_pshoot.push_back( pos_pshoot.back()+n+make_vec3(getVariation(0.02,generator),getVariation(0.01,generator), 0.f ) );
	
  	rad_pshoot.push_back(params.shoot_radius);
	
      }

      rad_pshoot.back()=0.01f;

      std::vector<vec3> tmp;
      tmp.resize(pos_pshoot.size());
      for( uint i=0; i<pos_pshoot.size(); i++ ){
  	tmp.at(i) = pos_pshoot.at(i) + origin;
      }
	  
      U = context->addTube(params.wood_subdivisions,tmp,rad_pshoot, params.wood_texture_file.c_str() );
      UUID_branch_plant.insert(UUID_branch_plant.end(), U.begin(), U.end() );

      //grape clusters
      std::vector<std::vector<uint> > UUID_grapes;
      if( params.grape_radius>0 && params.cluster_radius>0 ){
      
	float fgrape = fmax(0.1,0.2+getVariation(0.2,generator));
	vec3 p_grape = interpolateTube( tmp, fgrape );
	int sgn=1;
	if( unif_distribution(generator)<0.5 ){
	  sgn = -1;
	}
	vec3 offset(sgn*getVariation(0.1,generator)*sin(params.canopy_rotation),sgn*getVariation(0.1,generator)*cos(params.canopy_rotation),0.f);
	UUID_grapes = addGrapeCluster( p_grape+offset, params.grape_radius, params.cluster_radius );
      }
      UUID_fruit_plant.push_back( UUID_grapes );

      //leaves
      float flip = 0;
      if( unif_distribution(generator)<0.5 ){
	flip = 1;
      }
      float lfrac = 1.f;
      while( lfrac>0.*params.leaf_width ){

  	float lsize = fmaxf(params.leaf_width*(1.f-exp(-5.f*(1-lfrac))),0.1*params.leaf_width);
	
  	vec3 pos_leaf = interpolateTube( pos_pshoot, lfrac );

  	vec3 parent_normal = interpolateTube( pos_pshoot, fmax(0,lfrac-0.001) )-pos_leaf;
  	parent_normal.normalize();
  	vec3 leaf_offset = rotatePointAboutLine(make_vec3(0,lsize*(0.3+getVariation(0.25,generator)),0), make_vec3(0,0,0), parent_normal, flip*M_PI+unif_distribution(generator)*0.25*M_PI );

  	float s;
  	if( int(flip)%2==0 ){
  	  s = 1;
  	}else{
  	  s = -1;
  	}

  	float Rphi = -params.canopy_rotation - s*0.5*M_PI*(1.f+getVariation(0.4,generator));
  	float Rtheta = 0.4*M_PI*(1.f+getVariation(0.1,generator));

  	vec3 position = origin+pos_leaf+leaf_offset;

  	UUID_leaf_plant.push_back( context->addTile( make_vec3(0,0,0), make_vec2(lsize,lsize), make_SphericalCoord(0,0), params.leaf_subdivisions, params.leaf_texture_file.c_str() ) );
	
  	context->rotatePrimitive(UUID_leaf_plant.back(),-Rtheta,"y");
  	context->rotatePrimitive(UUID_leaf_plant.back(),Rphi,"z");
  	context->translatePrimitive(UUID_leaf_plant.back(),position);

	lfrac = lfrac - params.leaf_spacing_fraction*lsize*(1.f+getVariation(0.25,generator));
	
  	flip++;
			
      }
	    
    }
	
  }

  UUID_trunk.push_back( UUID_trunk_plant );
  UUID_branch.push_back( UUID_branch_plant );
  UUID_leaf.push_back( UUID_leaf_plant );
  UUID_fruit.push_back( UUID_fruit_plant );

}
