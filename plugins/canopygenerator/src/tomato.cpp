#include "CanopyGenerator.h"

using namespace helios;
using namespace std;

std::vector<std::vector<uint> > tomatoCluster( const vec3 center, const TomatoParameters &params, Context* context ){

  std::vector<std::vector<uint> > U;

  U.push_back( context->addSphere( params.fruit_subdivisions, center, params.fruit_radius, params.fruit_color ) );

  float azimuth = 2.f*PI_F*context->randu();

  if( context->randu()<0.75 ){

    U.push_back( context->addSphere( params.fruit_subdivisions, center+sphere2cart(make_SphericalCoord(1.5*params.fruit_radius,0,azimuth)), params.fruit_radius, params.fruit_color ) );

  }

  if( context->randu()<0.5 ){

    U.push_back( context->addSphere( params.fruit_subdivisions, center+sphere2cart(make_SphericalCoord(1.5*params.fruit_radius,0,azimuth+0.5*PI_F)), params.fruit_radius, params.fruit_color ) );

  }

  return U;
  

}

std::vector<uint> lateralLeaves( const float bfrac, const TomatoParameters params, const float leaf_scale, const std::vector<vec3> nodes, std::string texture, Context* context ){

  float downangle = 0.1*PI_F;
  
  vec3 r0 = interpolateTube( nodes, bfrac);
  vec3 r1 = interpolateTube( nodes, 1.01*bfrac);
  vec3 dr = r1-r0;

  float elevation = cart2sphere(dr).elevation;
  float azimuth = cart2sphere(dr).azimuth;

  std::vector<uint> U1, U2;
  
  U1 = context->addTile( make_vec3(0,0,0), make_vec2(leaf_scale,leaf_scale*0.5), make_SphericalCoord(0,PI_F), params.leaf_subdivisions, params.leaf_texture_file.c_str() );

  context->rotatePrimitive( U1, elevation, "x" );
  context->rotatePrimitive( U1, downangle, "y" );
  context->rotatePrimitive( U1, -azimuth, "z" );

  context->translatePrimitive( U1, r0 + 0.6*leaf_scale*make_vec3( cosf(-azimuth), sinf(-azimuth), -0.5*sinf(downangle)) );

  U2 = context->addTile( make_vec3(0,0,0), make_vec2(leaf_scale,leaf_scale*0.5), make_SphericalCoord(0,PI_F), params.leaf_subdivisions, params.leaf_texture_file.c_str() );

  context->rotatePrimitive( U2, elevation, "x" );
  context->rotatePrimitive( U2, downangle, "y" );
  context->rotatePrimitive( U2, -azimuth+PI_F, "z" );

  context->translatePrimitive( U2, r0 - 0.6*leaf_scale*make_vec3( cosf(-azimuth), sinf(-azimuth), 0.5*sinf(downangle)) );

  U1.insert( U1.end(), U2.begin(), U2.end() );

  return U1;
	     

}

void tomatoShoot( const TomatoParameters params, const helios::vec3 base_position, const helios::vec3 base_direction, const float length, const float bend_angle, std::vector<std::vector<uint> > &leaf_UUIDs,
                  std::vector<uint> &branch_UUIDs, std::vector<std::vector<std::vector<uint> > > &fruit_UUIDs, Context* context ){

  std::vector<vec3> nodes;
  std::vector<float> radius;
  std::vector<RGBcolor> color;

  int node_count = round( 2*length*params.shoot_subdivisions );

  vec3 dir = base_direction;
  dir.normalize();

  SphericalCoord base_angle = cart2sphere(dir);

  float shoot_radius = 0.005;

  nodes.push_back( base_position );
  radius.push_back( shoot_radius );
  color.push_back( params.shoot_color );

  float theta = base_angle.elevation;
  for( int i=1; i<node_count; i++ ){

    float vfrac = float(i)/float(node_count-1);

    radius.push_back( shoot_radius*params.plant_height*(1.f-0.8*vfrac) );

    vec3 position = nodes.back() + length/float(node_count-1)*(sphere2cart(make_SphericalCoord(1,theta,base_angle.azimuth)));

    theta -= bend_angle/float(node_count-1);

    nodes.push_back( position );

    color.push_back( params.shoot_color );

  }

  std::vector<uint> U = context->addTube( params.shoot_subdivisions, nodes, radius, color );
  branch_UUIDs.insert( branch_UUIDs.end(), U.begin(), U.end() );

  //tip leaf
  U = context->addTile( make_vec3(0,0,0), make_vec2(params.leaf_length,params.leaf_length*0.5), make_SphericalCoord(0,PI_F), params.leaf_subdivisions, params.leaf_texture_file.c_str() );

  context->rotatePrimitive( U, -theta, "y" );
  context->rotatePrimitive( U, -base_angle.azimuth+0.5*PI_F, "z" );

  context->translatePrimitive( U, nodes.back()+sphere2cart(make_SphericalCoord(0.45*params.leaf_length,theta,base_angle.azimuth)) );

  leaf_UUIDs.push_back(U);

  //primary lateral leaves

  leaf_UUIDs.push_back( lateralLeaves( 0.9, params, 0.3*params.leaf_length, nodes, params.leaf_texture_file, context ) );

  leaf_UUIDs.push_back( lateralLeaves( 0.8, params, 0.85*params.leaf_length, nodes, params.leaf_texture_file, context ) );

  leaf_UUIDs.push_back( lateralLeaves( 0.6, params, 0.3*params.leaf_length, nodes, params.leaf_texture_file, context ) );

  leaf_UUIDs.push_back( lateralLeaves( 0.45, params, 0.75*params.leaf_length, nodes, params.leaf_texture_file, context ) );

  //fruit

  if( params.fruit_radius>0 && context->randu()<0.66 ){

    vec3 cluster_position = interpolateTube( nodes, 0.25 ) - make_vec3(0,0,4*params.fruit_radius);

    fruit_UUIDs.push_back( tomatoCluster( cluster_position, params, context ) );

  }
    
}

uint CanopyGenerator::tomato(const TomatoParameters &params, const vec3 &origin ){

    std::vector<std::vector<uint> > leaf_UUIDs;
    std::vector<uint> branch_UUIDs;
    std::vector<std::vector<std::vector<uint> > > fruit_UUIDs;

  //main stem

  std::vector<vec3> nodes;
  std::vector<float> radius;
  std::vector<RGBcolor> color;

  int node_count = round( 2*params.plant_height*params.shoot_subdivisions );

  float stem_radius = 0.02;

  nodes.push_back( origin );
  radius.push_back( stem_radius );
  color.push_back( params.shoot_color );
  
  for( int i=1; i<node_count; i++ ){

    float vfrac = float(i)/float(node_count-1);

    radius.push_back( stem_radius*params.plant_height*(1.f-0.9*vfrac) );

    vec3 position = nodes.back() + params.plant_height/float(node_count-1)*make_vec3(0,0,1);
 
    nodes.push_back( position );

    color.push_back( params.shoot_color );

  }

  branch_UUIDs = context->addTube( params.shoot_subdivisions, nodes, radius, color );

  std::vector<float> shoot_heights;
  shoot_heights.push_back( 0.4 );
  shoot_heights.push_back( 0.5 );
  shoot_heights.push_back( 0.6 );
  shoot_heights.push_back( 0.65 );
  shoot_heights.push_back( 0.7 );
  shoot_heights.push_back( 0.75 );
  shoot_heights.push_back( 0.8 );
  shoot_heights.push_back( 0.85 );
  shoot_heights.push_back( 0.9 );
  shoot_heights.push_back( 0.95 );
  shoot_heights.push_back( 1 );

  int Nshoots = shoot_heights.size();

  for( int i=0; i<Nshoots; i++ ){

    vec3 position = interpolateTube( nodes, shoot_heights.at(i) );

    vec3 base_direction = sphere2cart( make_SphericalCoord( 0.2*PI_F+0.2*PI_F*float(i)/float(Nshoots-1), 2*PI_F*context->randu()) );

    float tip_angle = 0.5*PI_F+0.2*PI_F*float(i)/float(Nshoots-1);

    float length = (0.3+0.5*float(i)/float(Nshoots-1))*params.plant_height;

    tomatoShoot( params, position, base_direction, length, tip_angle, leaf_UUIDs, branch_UUIDs, fruit_UUIDs, context );

  }

  UUID_leaf.push_back( leaf_UUIDs );
  UUID_branch.push_back( branch_UUIDs );
  UUID_fruit.push_back( fruit_UUIDs );

  return UUID_leaf.size()-1;

}
