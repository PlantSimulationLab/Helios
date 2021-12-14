#include "CanopyGenerator.h"

using namespace helios;
using namespace std;

std::vector<std::vector<uint> > strawberryCluster(const vec3 center, const StrawberryParameters &params, Context* context ){


  std::vector<std::vector<uint> > U;

  std::vector<vec3> nodes;
  std::vector<float> radius;

  nodes.push_back( center + make_vec3(0,0,0.6)*params.fruit_radius );
  nodes.push_back( center + make_vec3(0,0,0.45)*params.fruit_radius );
  nodes.push_back( center + make_vec3(0,0,0.3)*params.fruit_radius );
  nodes.push_back( center + make_vec3(0,0,0.1)*params.fruit_radius );
  nodes.push_back( center + make_vec3(0,0,-0.2)*params.fruit_radius );
  nodes.push_back( center + make_vec3(0,0,-0.4)*params.fruit_radius );
  nodes.push_back( center + make_vec3(0,0,-0.5)*params.fruit_radius );

  radius.push_back( 1e-3*params.fruit_radius );
  radius.push_back( 0.4*params.fruit_radius );
  radius.push_back( 0.5*params.fruit_radius );
  radius.push_back( 0.45*params.fruit_radius );
  radius.push_back( 0.2*params.fruit_radius );
  radius.push_back( 0.15*params.fruit_radius );
  radius.push_back( 1e-3*params.fruit_radius );

  U.push_back(  context->addTube( params.fruit_subdivisions, nodes, radius, params.fruit_texture_file.c_str() ) );

  float azimuth = 2.f*M_PI*context->randu();

  if( context->randu()<0.75 ){

    std::vector<vec3> nodes2 = nodes;
    for( int i=0; i<nodes2.size(); i++ ){
      nodes2.at(i) = nodes2.at(i) = sphere2cart(make_SphericalCoord(1.5*params.fruit_radius,0,azimuth));
    }

    U.push_back(  context->addTube( params.fruit_subdivisions, nodes2, radius, params.fruit_texture_file.c_str() ) );

  }

  if( context->randu()<0.5 ){

    std::vector<vec3> nodes3 = nodes;
    for( int i=0; i<nodes3.size(); i++ ){
      nodes3.at(i) = nodes3.at(i) = sphere2cart(make_SphericalCoord(1.5*params.fruit_radius,0,azimuth+0.5*M_PI));
    }

    U.push_back(  context->addTube( params.fruit_subdivisions, nodes3, radius, params.fruit_texture_file.c_str() ) );

  }

  return U;
  

}

std::vector<uint> leafPrototype(const StrawberryParameters &params, Context* context ){

  int Nx = params.leaf_subdivisions.x;
  int Ny = ceil( params.leaf_subdivisions.y*0.5 );

  float dx = 1.f/float(Nx);
  float dy = 0.5f/float(Ny);

  std::vector<uint> UUIDs;

  float a0 = 0.6;
  float e0 = 0.4f;

  for( int i=0; i<Nx; i++ ){
    for( int j=0; j<Ny; j++ ){

      float x = i*dx;
      float y = j*dy;

      float mag, z;

      mag = sqrt( x*x + 2*y*y );
      z = a0*mag/(e0+mag);
      vec3 v0( x, y, z );

      mag = sqrt( (x+dx)*(x+dx) + 2*y*y );
      z = a0*mag/(e0+mag);
      vec3 v1( x+dx, y, z );

      mag = sqrt( (x+dx)*(x+dx) + 2*(y+dy)*(y+dy) );
      z = a0*mag/(e0+mag);
      vec3 v2( x+dx, y+dy, z );

      mag = sqrt( x*x + 2*(y+dy)*(y+dy) );
      z = a0*mag/(e0+mag);
      vec3 v3( x, y+dy, z );

        vec2 uv0( 1.f-x, 0.5+j*dy);
        vec2 uv1( 1.f-x-dx, 0.5+j*dy );
        vec2 uv2( 1.f-x-dx, 0.5+(j+1)*dy );
        vec2 uv3( 1.f-x, 0.5+(j+1)*dy );

      UUIDs.push_back( context->addTriangle( v0, v1, v2, params.leaf_texture_file.c_str(), uv0, uv1, uv2 ) );
      UUIDs.push_back( context->addTriangle( v0, v2, v3, params.leaf_texture_file.c_str(), uv0, uv2, uv3 ) );

      v0.y = -v0.y;
      v1.y = -v1.y;
      v2.y = -v2.y;
      v3.y = -v3.y;

        uv0 = make_vec2( 1.f-x, 0.5-j*dy);
        uv1 = make_vec2( 1.f-x-dx, 0.5-j*dy );
        uv2 = make_vec2( 1.f-x-dx, 0.5-(j+1)*dy );
        uv3 = make_vec2( 1.f-x, 0.5-(j+1)*dy );

      UUIDs.push_back( context->addTriangle( v0, v2, v1, params.leaf_texture_file.c_str(), uv0, uv2, uv1 ) );
      UUIDs.push_back( context->addTriangle( v0, v3, v2, params.leaf_texture_file.c_str(), uv0, uv3, uv2 ) );
      
    }
  }

  return UUIDs;

}

void
strawberryShoot(const StrawberryParameters &params, const helios::vec3 base_position, const helios::vec3 base_direction,
                const float length, const float bend_angle, const std::vector<uint> &leaf_prototype,
                std::vector<std::vector<uint> > &leaf_UUIDs,
                std::vector<uint> &branch_UUIDs, std::vector<std::vector<std::vector<uint> > > &fruit_UUIDs, Context *context) {

    std::vector<uint> U;

  std::vector<vec3> nodes;
  std::vector<float> radius;
  std::vector<RGBcolor> color;

  int node_count = round( 2*length*params.stem_subdivisions );

  vec3 dir = base_direction;
  dir.normalize();

  SphericalCoord base_angle = cart2sphere(dir);

  nodes.push_back( base_position );
  radius.push_back( params.stem_radius );
  color.push_back( params.stem_color );

  float theta = base_angle.elevation;
  for( int i=1; i<node_count; i++ ){

    float vfrac = float(i)/float(node_count-1);

    radius.push_back( params.stem_radius*params.plant_height*(1.f-0.8*vfrac) );

    vec3 position = nodes.back() + length/float(node_count-1)*(sphere2cart(make_SphericalCoord(1,theta,base_angle.azimuth)));

    theta -= bend_angle/float(node_count-1);

    nodes.push_back( position );

    color.push_back( params.stem_color );

  }

  U = context->addTube( params.stem_subdivisions, nodes, radius, color );
  branch_UUIDs.insert( branch_UUIDs.end(), U.begin(), U.end() );

  //tip leaf


  U = context->copyPrimitive( leaf_prototype );

  context->scalePrimitive( U, make_vec3( params.leaf_length, 0.8*params.leaf_length, params.leaf_length ) );
  context->rotatePrimitive( U, 0.75*(0.6*M_PI-theta), "y" );
  context->rotatePrimitive( U, -base_angle.azimuth+0.5*M_PI, "z" );

  context->translatePrimitive( U, nodes.back() );

  leaf_UUIDs.push_back(U);

  //lateral leaves
  U = context->copyPrimitive( leaf_prototype );

  context->scalePrimitive( U, 0.8*make_vec3( params.leaf_length, 0.8*params.leaf_length, params.leaf_length ) );
  context->rotatePrimitive( U, 0.75*(0.6*M_PI-theta), "y" );
  context->rotatePrimitive( U, -base_angle.azimuth-0.05*M_PI, "z" );

  context->translatePrimitive( U, nodes.back() );

  leaf_UUIDs.push_back(U);

  U = context->copyPrimitive( leaf_prototype );

  context->scalePrimitive( U, 0.8*make_vec3( params.leaf_length, 0.8*params.leaf_length, params.leaf_length ) );
  context->rotatePrimitive( U, 0.75*(0.6*M_PI-theta), "y" );
  context->rotatePrimitive( U, -base_angle.azimuth+1.05*M_PI, "z" );

  context->translatePrimitive( U, nodes.back() );

  leaf_UUIDs.push_back(U);

  //fruit

  if( context->randu()<params.clusters_per_stem ){

    vec3 cluster_position = interpolateTube( nodes, 0.35 )+ sphere2cart( make_SphericalCoord( 0.1+0.2*context->randu(), 0, 2*M_PI*context->randu() ) );

    fruit_UUIDs.push_back(strawberryCluster( cluster_position, params, context ) );

  }
    
}

void CanopyGenerator::strawberry(const StrawberryParameters &params, const vec3 &origin ){

    std::vector<std::vector<uint> > leaf_UUIDs;
    std::vector<uint> branch_UUIDs;
    std::vector<std::vector<std::vector<uint> > > fruit_UUIDs;

  std::vector<uint> leaf_prototype = leafPrototype( params, context );

  for( int i=0; i<params.stems_per_plant; i++ ){

    vec3 position = origin + sphere2cart( make_SphericalCoord(0.1*context->randu()*params.plant_height,0,2*M_PI*context->randu()) ) ;
    
    vec3 base_direction = sphere2cart( make_SphericalCoord( 0.45*M_PI-0.15*M_PI*float(i)/float(params.stems_per_plant-1), 2*M_PI*context->randu()) );

    float tip_angle = 0.3*M_PI*float(i)/float(params.stems_per_plant-1);

    float length = (0.75+0.25*float(params.stems_per_plant-1-i)/float(params.stems_per_plant-1))*params.plant_height*(0.9+0.2*context->randu());

      strawberryShoot(params, position, base_direction, length, tip_angle, leaf_prototype, leaf_UUIDs, branch_UUIDs, fruit_UUIDs, context);

  }

  context->deletePrimitive( leaf_prototype );

  UUID_leaf.push_back( leaf_UUIDs );
  UUID_branch.push_back( branch_UUIDs );
  UUID_fruit.push_back( fruit_UUIDs );
  

}
