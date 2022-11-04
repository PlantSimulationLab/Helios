#include "CanopyGenerator.h"

using namespace helios;
using namespace std;

std::vector<std::vector<uint> > beanCluster( const vec3 center, const BeanParameters params, Context* context ){

    std::vector<std::vector<uint> > U;

    //todo, nothing here yet

    return U;


}

std::vector<uint> leafPrototype( const BeanParameters &params, Context *context ){

  int Nx = params.leaf_subdivisions.x;
  int Ny = ceil( params.leaf_subdivisions.y*0.5 );

  float dx = 1.f/float(Nx);
  float dy = 0.5f/float(Ny);

  std::vector<uint> UUIDs;//(Nx*Ny*4);

  float a0 = 0.6; //magnitude of leaf bend
  float e0 = 0.5f;  //rate of leaf bend

  int ii=0;
  for( int i=0; i<Nx; i++ ){
    for( int j=0; j<Ny; j++ ){

      float x = float(i)*dx;
      float y = float(j)*dy;

      float mag, z;

      z = a0*x/(e0+x) - e0*y*y;
      vec3 v0( x, y, z );

      z = a0*(x+dx)/(e0+x+dx) - e0*y*y;
      vec3 v1( x+dx, y, z );

      z = a0*(x+dx)/(e0+x+dx) - e0*(y+dy)*(y+dy);
      vec3 v2( x+dx, y+dy, z );

      z = a0*x/(e0+x) - e0*(y+dy)*(y+dy);
      vec3 v3( x, y+dy, z );

      vec2 uv0( x, 0.5f+j*dy);
      vec2 uv1( x+dx, 0.5f+j*dy );
      vec2 uv2( x+dx, 0.5f+(j+1)*dy );
      vec2 uv3( x, 0.5f+(j+1)*dy );

      UUIDs.push_back( context->addTriangle( v0, v1, v2, params.leaf_texture_file.c_str(), uv0, uv1, uv2 ) );
      UUIDs.push_back( context->addTriangle( v0, v2, v3, params.leaf_texture_file.c_str(), uv0, uv2, uv3 ) );

      v0.y = -v0.y;
      v1.y = -v1.y;
      v2.y = -v2.y;
      v3.y = -v3.y;

      uv0 = make_vec2( x, 0.5f-j*dy);
      uv1 = make_vec2( x+dx, 0.5f-j*dy );
      uv2 = make_vec2( x+dx, 0.5f-(j+1)*dy );
      uv3 = make_vec2( x, 0.5f-(j+1)*dy );

      UUIDs.push_back( context->addTriangle( v0, v2, v1, params.leaf_texture_file.c_str(), uv0, uv2, uv1 ) );
      UUIDs.push_back( context->addTriangle( v0, v3, v2, params.leaf_texture_file.c_str(), uv0, uv3, uv2 ) );


      ii+=4;
    }
  }

  context->rotatePrimitive( UUIDs, 0.6, "y" );
  context->translatePrimitive( UUIDs, make_vec3(0.1, 0.12, 0.06) );

//    std::vector<uint> UUIDs = context->loadOBJ( "../obj/leaf_new.obj", make_vec3(0.5,0,0), 0, nullrotation, RGB::green, true );
//    context->scalePrimitive( UUIDs, make_vec3(0.4,0.5,0.5) );
//    context->translatePrimitive( UUIDs, make_vec3(-0.25,0.6,0.06) );
//    context->rotatePrimitive( UUIDs, -0.5*M_PI, "z" );

//    std::vector<uint> UUIDs = context->addTile( make_vec3(0.5,0,0), make_vec2(1,1), nullrotation, make_int2(1,1), params.leaf_texture_file.c_str() );

    std::vector<vec3> nodes;
    nodes.push_back( make_vec3(0,0,0) );
    nodes.push_back( make_vec3(0.06, 0, 0.0 ) );
    nodes.push_back( make_vec3(0.13, 0, 0.02) );
    nodes.push_back( make_vec3(0.13, 0, 0.05) );

    float Rpetiole = 0.015;

    std::vector<float> radius{Rpetiole, Rpetiole, Rpetiole*0.9f, Rpetiole*0.8f};
    std::vector<RGBcolor> color{params.shoot_color,params.shoot_color,params.shoot_color,params.shoot_color};

    std::vector<uint> U = context->addTube( params.shoot_subdivisions, nodes, radius, color );

    UUIDs.insert( UUIDs.end(), U.begin(), U.end() );

    return UUIDs;

}

void beanLeaflet( const BeanParameters &params, const helios::vec3 base_position, const helios::vec3 base_direction, float length, float bend_angle, float parent_radius, std::vector<std::vector<uint> > &leaf_UUIDs,
                  std::vector<uint> &branch_UUIDs, std::vector<std::vector<std::vector<uint> > > &fruit_UUIDs, std::vector<uint> &leaf_prototype, std::minstd_rand0 &rand_generator, Context* context ){

    std::vector<vec3> nodes;
    std::vector<float> radius;
    std::vector<RGBcolor> color;

    int node_count = std::ceil( 2.f*length*params.shoot_subdivisions )+1;

    vec3 dir = base_direction;
    dir.normalize();

    SphericalCoord base_angle = cart2sphere(dir);

    nodes.push_back( base_position );
    radius.push_back( 0.5f*parent_radius );
    color.push_back( params.shoot_color );

    float theta = base_angle.elevation;
    for( int i=1; i<node_count; i++ ){

        float vfrac = float(i)/float(node_count-1);

        radius.push_back( 0.45f*parent_radius*(1.f-0.3*vfrac) );

        vec3 position = nodes.back() + length/float(node_count-1)*(sphere2cart(make_SphericalCoord(1,theta,base_angle.azimuth)));

        theta -= bend_angle/float(node_count-1);

        nodes.push_back( position );

        color.push_back( params.shoot_color );

    }

    std::vector<uint> U = context->addTube( params.shoot_subdivisions, nodes, radius, color );
    branch_UUIDs.insert( branch_UUIDs.end(), U.begin(), U.end() );

    //tip leaf
    U = context->copyPrimitive( leaf_prototype );

    context->scalePrimitive( U, make_vec3(params.leaf_length, params.leaf_length*0.9f, params.leaf_length) );
    context->rotatePrimitive( U, -(theta + getVariation( 0.3*M_PI, rand_generator )), "y" );
    context->rotatePrimitive( U, -base_angle.azimuth+0.5f*M_PI, "z" );

    context->translatePrimitive( U, nodes.back()-sphere2cart(make_SphericalCoord(0.001f,theta,base_angle.azimuth)) );

    leaf_UUIDs.push_back(U);

    //lateral leaves

    float leaf_downangle = 0.1*M_PI;
    float leaf_bfrac = 0.8;

    vec3 r0 = interpolateTube( nodes, leaf_bfrac);
    vec3 r1 = interpolateTube( nodes, 1.01*leaf_bfrac);
    vec3 dr = r1-r0; //vector pointing in direction of shoot at location of leaf

    float elevation = cart2sphere(dr).elevation;
    float azimuth = cart2sphere(dr).azimuth;

    std::vector<uint> U1, U2;

    U1 = context->copyPrimitive( leaf_prototype );

    context->scalePrimitive( U1, make_vec3(params.leaf_length, params.leaf_length*0.9, params.leaf_length) );
    context->rotatePrimitive( U1, elevation + getVariation( 0.2*M_PI, rand_generator ), "x" );
    context->rotatePrimitive( U1, leaf_downangle + getVariation( 0.2*M_PI, rand_generator ), "y" );
    context->rotatePrimitive( U1, -azimuth, "z" );

    context->translatePrimitive( U1, r0 + 0.01*params.leaf_length*make_vec3( cosf(-azimuth), sinf(-azimuth), 0) );

    U2 = context->copyPrimitive( leaf_prototype );

    context->scalePrimitive( U2, make_vec3(params.leaf_length, params.leaf_length*0.9, params.leaf_length) );
    context->rotatePrimitive( U2, -elevation + getVariation( 0.2*M_PI, rand_generator ), "x" );
    context->rotatePrimitive( U2, leaf_downangle + getVariation( 0.2*M_PI, rand_generator ), "y" );
    context->rotatePrimitive( U2, -azimuth+M_PI, "z" );

    context->translatePrimitive( U2, r0 - 0.01*params.leaf_length*make_vec3( cosf(-azimuth), sinf(-azimuth), 0)  );

    leaf_UUIDs.push_back(U1);
    leaf_UUIDs.push_back(U2);

    //fruit

    if( params.pod_length>0 && context->randu()<0.66 ){

        vec3 cluster_position = interpolateTube( nodes, 0.25 ) - make_vec3(0,0,4*params.pod_length);

        fruit_UUIDs.push_back( beanCluster( cluster_position, params, context ) );

    }

}

void beanShoot( const BeanParameters &params, const helios::vec3 base_position, const helios::SphericalCoord &shoot_direction, float scale, std::vector<std::vector<uint> > &leaf_UUIDs,
                std::vector<uint> &branch_UUIDs, std::vector<std::vector<std::vector<uint> > > &fruit_UUIDs, std::vector<uint> &leaf_prototype, std::minstd_rand0 &rand_generator, Context* context ){

    std::vector<uint> UUIDs_stem;
    std::vector<std::vector<uint> >  UUIDs_leaf;
    std::vector<std::vector<std::vector<uint> > > UUIDs_pod;

    std::vector<vec3> nodes;

    nodes.push_back( base_position + make_vec3(0,0,0)*scale );
    nodes.push_back( base_position + make_vec3(0,0,params.stem_length/2.f)*scale );
    nodes.push_back( base_position + make_vec3(0,0,params.stem_length)*scale );

    std::vector<float> radius{params.stem_radius*scale,params.stem_radius*scale,params.stem_radius*scale*0.9f};
    std::vector<RGBcolor> color{params.shoot_color,params.shoot_color,params.shoot_color};

    UUIDs_stem = context->addTube( params.shoot_subdivisions, nodes, radius, color );

  std::vector<uint> U = context->copyPrimitive( leaf_prototype );

  float az0 = getVariation( 2.f*M_PI, rand_generator );

  context->scalePrimitive( U, make_vec3(params.leaf_length, params.leaf_length*0.9f, params.leaf_length)*0.7 );
  context->rotatePrimitive( U, getVariation( 0.2*M_PI, rand_generator ), "y" );
  context->rotatePrimitive( U, az0, "z" );
  context->translatePrimitive( U, base_position+make_vec3(0,0, 0.75f*params.stem_length)*scale );

  leaf_UUIDs.push_back( U );

  U = context->copyPrimitive( leaf_prototype );

  context->scalePrimitive( U, make_vec3(params.leaf_length, params.leaf_length*0.9f, params.leaf_length)*0.7 );
  context->rotatePrimitive( U, getVariation( 0.2*M_PI, rand_generator ), "y" );
  context->rotatePrimitive( U, az0+M_PI, "z" );
  context->translatePrimitive( U, base_position+make_vec3(0,0, 0.75f*params.stem_length)*scale );

  leaf_UUIDs.push_back( U );

    float shoot_zenith = 0.2*M_PI+getVariation(0.1*M_PI, rand_generator );
    float shoot_azimuth = getVariation( 2.f*M_PI, rand_generator );

    vec3 leaflet_direction = sphere2cart(  make_SphericalCoord(0.5f*M_PI-shoot_zenith, shoot_azimuth + getVariation( 0.2f*M_PI, rand_generator )) );

    beanLeaflet( params, nodes.back(), leaflet_direction, params.leaflet_length, 0.2*M_PI, params.stem_radius, UUIDs_leaf, UUIDs_stem, UUIDs_pod, leaf_prototype, rand_generator, context );

    leaflet_direction = sphere2cart(  make_SphericalCoord(0.5f*M_PI-shoot_zenith, M_PI+shoot_azimuth + getVariation( 0.2f*M_PI, rand_generator )) );

    beanLeaflet( params, nodes.back(), leaflet_direction, params.leaflet_length, 0.2*M_PI, params.stem_radius, UUIDs_leaf, UUIDs_stem, UUIDs_pod, leaf_prototype, rand_generator, context );

    for( uint i=0; i<UUIDs_leaf.size(); i++ ) {
        context->rotatePrimitive(UUIDs_leaf.at(i), shoot_direction.zenith, "x");
        context->rotatePrimitive(UUIDs_leaf.at(i), shoot_direction.azimuth, "z");
    }
    context->rotatePrimitive(UUIDs_stem, shoot_direction.zenith, "x");
    context->rotatePrimitive(UUIDs_stem, shoot_direction.azimuth, "z");

    leaf_UUIDs.insert( leaf_UUIDs.end(), UUIDs_leaf.begin(), UUIDs_leaf.end() );
    branch_UUIDs.insert( branch_UUIDs.end(), UUIDs_stem.begin(), UUIDs_stem.end() );
    fruit_UUIDs.insert( fruit_UUIDs.end(), UUIDs_pod.begin(), UUIDs_pod.end() );

}

uint CanopyGenerator::bean(const BeanParameters &params, const vec3 &origin ){

    std::vector<std::vector<uint> > leaf_UUIDs;
    std::vector<uint> branch_UUIDs;
    std::vector<std::vector<std::vector<uint> > > fruit_UUIDs;

    std::vector<uint> leaf_prototype = leafPrototype(params, context);

    beanShoot( params, origin, make_SphericalCoord(0.5*M_PI,0), 1, leaf_UUIDs, branch_UUIDs, fruit_UUIDs, leaf_prototype, generator, context );

    UUID_leaf.push_back( leaf_UUIDs );
    UUID_branch.push_back( branch_UUIDs );
    UUID_fruit.push_back( fruit_UUIDs );

    context->deletePrimitive( leaf_prototype );

    return UUID_leaf.size()-1;

}
