#include "CanopyGenerator.h"

using namespace helios;
using namespace std;


std::vector<uint> leafPrototype( const WalnutCanopyParameters params, std::minstd_rand0 generator, Context* context ){

  int Nx = params.leaf_subdivisions.x;
  int Ny = ceil( params.leaf_subdivisions.y*0.5 );

  float dx = 1.f/float(Nx);
  float dy = 0.5f/float(Ny);

  std::uniform_real_distribution<float> unif_distribution;

  std::vector<uint> UUIDs;

  float fold = 0.1*M_PI;//unif_distribution(generator)*0.3*M_PI;

  for( int i=0; i<Nx; i++ ){
    for( int j=0; j<Ny; j++ ){

      float x = i*dx;
      float y = j*dy;

      float mag, z;

      mag = sqrt( x*x + 2*y*y );
      z = y*sinf( fold );
      vec3 v0( x, y, z );

      mag = sqrt( (x+dx)*(x+dx) + 2*y*y );
      z = y*sinf( fold );
      vec3 v1( x+dx, y, z );

      mag = sqrt( (x+dx)*(x+dx) + 2*(y+dy)*(y+dy) );
      z = (y+dy)*sinf( fold );
      vec3 v2( x+dx, y+dy, z );

      mag = sqrt( x*x + 2*(y+dy)*(y+dy) );
      z = (y+dy)*sinf( fold );
      vec3 v3( x, y+dy, z );

      vec2 uv0( x, 0.5+j*dy);
      vec2 uv1( x+dx, 0.5+j*dy );
      vec2 uv2( x+dx, 0.5+(j+1)*dy );
      vec2 uv3( x, 0.5+(j+1)*dy );

      UUIDs.push_back( context->addTriangle( v0, v1, v2, params.leaf_texture_file.c_str(), uv0, uv1, uv2 ) );
      UUIDs.push_back( context->addTriangle( v0, v2, v3, params.leaf_texture_file.c_str(), uv0, uv2, uv3 ) );

      v0.y = -v0.y;
      v1.y = -v1.y;
      v2.y = -v2.y;
      v3.y = -v3.y;

      uv0 = make_vec2( x, 0.5-j*dy);
      uv1 = make_vec2( x+dx, 0.5-j*dy );
      uv2 = make_vec2( x+dx, 0.5-(j+1)*dy );
      uv3 = make_vec2( x, 0.5-(j+1)*dy );

      UUIDs.push_back( context->addTriangle( v0, v2, v1, params.leaf_texture_file.c_str(), uv0      , uv2, uv1 ) );
      UUIDs.push_back( context->addTriangle( v0, v3, v2, params.leaf_texture_file.c_str(), uv0, uv3, uv2 ) );
      
    }
  }

  return UUIDs;

}

std::vector<uint> nutPrototype( const WalnutCanopyParameters params, Context* context ){

  std::vector<uint> UUIDs = context->addSphere( params.fruit_subdivisions, make_vec3(0,0,0), 1, params.fruit_texture_file.c_str() );

  context->scalePrimitive( UUIDs, make_vec3(0.85,0.85,1) );

  return UUIDs;

}

uint addBranch( const vec3 p_start, const vec3 tan_start, const float r_start, const vec3 p_end, const vec3 tan_end, const float r_end, const uint Nsegs, const WalnutCanopyParameters params, std::minstd_rand0 generator, Context* context ){

  uint ID;

  std::vector<vec3> position;
  std::vector<float> radius;

  // int ikink = randu(1,Nsegs-2);

  // vec3 pkink = spline_interp3( float(ikink)/float(Nsegs-1), p_start, tan_start, p_end, tan_end );
  // vec3 tankink = 0.5*(tan_start+tan_end)+0.*make_vec3(getVariation(1.f,generator),getVariation(1.f,generator),getVariation(1.f,generator));

  // for( int i=0; i<ikink; i++ ){

  //   position.push_back( spline_interp3( float(i)/float(Nsegs-1), p_start, tan_start, pkink, tankink ) );

  //   radius.push_back( r_start + float(i)/float(Nsegs-1)*(r_end-r_start) );

  // }

  // for( int i=ikink; i<Nsegs; i++ ){

  //   position.push_back( spline_interp3( float(i)/float(Nsegs-1), pkink, tankink, p_end, tan_end ) );

  //   radius.push_back( r_start + float(i)/float(Nsegs-1)*(r_end-r_start) );

  // }

  for( int i=0; i<Nsegs; i++ ){

    position.push_back( spline_interp3( float(i)/float(Nsegs-1), p_start, tan_start, p_end, tan_end ) );

    radius.push_back( r_start + float(i)/float(Nsegs-1)*(r_end-r_start) );

  }

  radius.front() *= 1.3;

  ID = context->addTubeObject(params.wood_subdivisions,position,radius, params.wood_texture_file.c_str() );

  return ID;

  
}

void branchRecursion( const std::vector<vec3> position_parent, const std::vector<float> radius_parent, uint level, uint max_levels, const std::vector<uint> leaf_prototype, const std::vector<uint> nut_prototype, std::vector<uint> &UUID_branch_plant, std::vector<std::vector<uint> > &UUID_leaf_plant, std::vector<std::vector<uint> > &UUID_fruit_plant, const WalnutCanopyParameters params, std::minstd_rand0 generator, Context* context ){

  std::uniform_real_distribution<float> unif_distribution;

  if( level>max_levels ){ //leaves and nuts

    int Nleaves = 6;

    for( int i=0; i<Nleaves; i++ ) {

        float u = 0.3 + 0.7 * float(i) / float(Nleaves - 1);

        float downangle = 0.15 * M_PI + getVariation(0.1 * M_PI, generator);

        vec3 r0 = interpolateTube(position_parent, u);
        vec3 r1 = interpolateTube(position_parent, 0.98 * u);
        vec3 dr = r1 - r0;

        //float elevation = cart2sphere(dr).elevation;
        float azimuth = cart2sphere(dr).azimuth + getVariation(0.2 * M_PI, generator);

        float elevation = 0.25 * M_PI + getVariation(0.1 * M_PI, generator);

        std::vector<uint> UUIDs = context->copyPrimitive(leaf_prototype);
        UUID_leaf_plant.push_back(UUIDs);

        context->scalePrimitive(UUIDs, make_vec3(params.leaf_length, 0.5 * params.leaf_length, 0.5 * params.leaf_length));

        context->rotatePrimitive(UUIDs, elevation, "x");
        context->rotatePrimitive(UUIDs, downangle, "y");
        context->rotatePrimitive(UUIDs, -azimuth, "z");

        context->translatePrimitive(UUIDs, r0 + 0.1 * params.leaf_length * make_vec3(cosf(-azimuth), sinf(-azimuth), -0.5 * sinf(downangle)));

        UUIDs = context->copyPrimitive(leaf_prototype);
        UUID_leaf_plant.push_back(UUIDs);

        context->scalePrimitive(UUIDs, make_vec3(params.leaf_length, 0.5 * params.leaf_length, 0.5 * params.leaf_length));

        context->rotatePrimitive(UUIDs, elevation, "x");
        context->rotatePrimitive(UUIDs, downangle, "y");
        context->rotatePrimitive(UUIDs, -azimuth + M_PI, "z");

        context->translatePrimitive(UUIDs, r0 - 0.1 * params.leaf_length * make_vec3(cosf(-azimuth), sinf(-azimuth), -0.5 * sinf(downangle)));

        if (i == Nleaves - 1) { //tip leaf

            UUIDs = context->copyPrimitive(leaf_prototype);
            UUID_leaf_plant.push_back(UUIDs);

            context->scalePrimitive(UUIDs, make_vec3(params.leaf_length, 0.5 * params.leaf_length, 0.5 * params.leaf_length));

            elevation = 0.25 * M_PI + getVariation(0.1 * M_PI, generator);

            context->rotatePrimitive(UUIDs, elevation, "y");
            context->rotatePrimitive(UUIDs, 0, "x");
            context->rotatePrimitive(UUIDs, -azimuth + 0.5 * M_PI, "z");

            context->translatePrimitive(UUIDs, r0 - 0. * params.leaf_length * make_vec3(sinf(-azimuth), cosf(-azimuth), 0));

        }

        //nut

        if (params.fruit_radius > 0) {

            float Rnut = unif_distribution(generator);

            if (Rnut < 0.005) {// triplet

                //uint objID = context->copyObject( nut_prototype );
                //UUID_fruit_plant.push_back( context->getObjectPointer(objID)->getPrimitiveUUIDs() );

                //context->getSphereObjectPointer(objID)->scale( params.fruit_radius );

                //context->getObjectPointer(objID)->translate( r0 - params.fruit_radius*make_vec3( sinf(-azimuth), cosf(-azimuth), -0.65) );

                std::vector<uint> nut_UUIDs = context->copyPrimitive(nut_prototype);
                UUID_fruit_plant.push_back(nut_UUIDs);

                context->scalePrimitive(nut_UUIDs, make_vec3(params.fruit_radius, params.fruit_radius, params.fruit_radius));

                context->translatePrimitive(nut_UUIDs, r0 - params.fruit_radius * make_vec3(sinf(-azimuth), cosf(-azimuth), -0.65));

            }
            if (Rnut < 0.01) { //double

                std::vector<uint> nut_UUIDs = context->copyPrimitive(nut_prototype);
                UUID_fruit_plant.push_back(nut_UUIDs);

                context->scalePrimitive(nut_UUIDs, make_vec3(params.fruit_radius, params.fruit_radius, params.fruit_radius));

                context->translatePrimitive(nut_UUIDs, r0 - params.fruit_radius * make_vec3(2.f * sinf(-azimuth), 2.f * cosf(-azimuth), -0.85));

            }
            if (Rnut < 0.05) { //single

                std::vector<uint> nut_UUIDs = context->copyPrimitive(nut_prototype);
                UUID_fruit_plant.push_back(nut_UUIDs);

                context->scalePrimitive(nut_UUIDs, make_vec3(params.fruit_radius, params.fruit_radius, params.fruit_radius));

                context->translatePrimitive(nut_UUIDs, r0 - params.fruit_radius * make_vec3(sinf(-azimuth), cosf(-azimuth), -0.75));


            }


        }

    }

    
  }else{ //branches

  float L = 0;
  int Nbranch;
  if( level==1 ){
    L = params.branch_length.x;
    Nbranch = 8;
  }else if( level==2 ){
    L = params.branch_length.y;
    Nbranch = 6;
  }else if( level==3 ){
    L = params.branch_length.z;
    Nbranch = 4;
  }

  if( L==0 ){
    return;
  }

  //int Nbranch = fmin(8,ceil(5*L));

  //random azimuthal rotation of first new branch
  float phi0 = 2*M_PI*unif_distribution(generator);

  for( int i=0; i<Nbranch; i++ ){

    //phi0 += 0.45*M_PI*(1.f+getVariation(0.2,generator));
    phi0 += 60*M_PI/180.f*(1.f+getVariation(0.3,generator));

    //new branch height as fraction of trunk height
    float u = fmin(1.f, 0.2+0.8*float(i+1)/float(Nbranch)+getVariation(0.05,generator));

    //position of new branch base
    vec3 pbase = interpolateTube( position_parent, u );
    //vector pointing in direction of parent branch
    vec3 nbase = pbase-interpolateTube( position_parent, 0.98*u );

    //unit vector perpendicular to parent branch
    vec3 bnorm = cross( make_vec3(0,1,1), nbase );
    bnorm.normalize();

    //vector perpendicular to new branch
    vec3 pvec1 = cross( bnorm, nbase );

    //direction of new branch base - apply random rotation relative to parent
    bnorm = rotatePointAboutLine( bnorm, make_vec3(0,0,0), pvec1, 0.1*M_PI+0.15*M_PI*float(i)/float(Nbranch-1) );

    //direction of new branch base - apply random rotation about parent
    vec3 bnorm1 = rotatePointAboutLine( bnorm, make_vec3(0,0,0), nbase, phi0 );
    bnorm1.normalize();
    vec3 bnorm2 = rotatePointAboutLine( bnorm, make_vec3(0,0,0), nbase, phi0+M_PI+getVariation(0.5,generator) );
    bnorm2.normalize();

    float L1 = (0.5+0.5*unif_distribution(generator))*L;
    float L2 = (0.5+0.5*unif_distribution(generator))*L;

    //base radius of new branch
    float rbase1 = 0.6*interpolateTube( radius_parent, u );
    float rbase2 = rbase1*(0.4+0.6*L2/L);
    rbase1 = rbase1*(0.4+0.6*L1/L);

    vec3 pend1 = bnorm1 + 0.2*L1*make_vec3(0,0,1);
    pend1.normalize();
    vec3 pend2 = bnorm2 + 0.2*L2*make_vec3(0,0,1);
    pend2.normalize();

    uint ID1 = addBranch( pbase, 0.5*L1*bnorm1, rbase1, pbase+pend1*L1, L1*make_vec3(0,0,1.f+getVariation(1,generator)), 0.15*rbase1, 8, params, generator, context );
    uint ID2 = addBranch( pbase, 0.5*L2*bnorm2, rbase2, pbase+pend2*L2, L2*make_vec3(0,0,1.f+getVariation(1,generator)), 0.15*rbase2, 8, params, generator, context );

    std::vector<uint> UUID = context->getObjectPointer(ID1)->getPrimitiveUUIDs();
    UUID_branch_plant.insert( UUID_branch_plant.end(), UUID.begin(), UUID.end() );
    UUID = context->getObjectPointer(ID2)->getPrimitiveUUIDs();
    UUID_branch_plant.insert( UUID_branch_plant.end(), UUID.begin(), UUID.end() );

    std::vector<vec3> pos_parent = context->getTubeObjectPointer(ID1)->getNodes();
    std::vector<float> rad_parent = context->getTubeObjectPointer(ID1)->getNodeRadii();

    branchRecursion( pos_parent, rad_parent, level+1, max_levels, leaf_prototype, nut_prototype, UUID_branch_plant, UUID_leaf_plant, UUID_fruit_plant, params, generator, context );

    pos_parent = context->getTubeObjectPointer(ID2)->getNodes();
    rad_parent = context->getTubeObjectPointer(ID2)->getNodeRadii();

    branchRecursion( pos_parent, rad_parent, level+1, max_levels, leaf_prototype, nut_prototype, UUID_branch_plant, UUID_leaf_plant, UUID_fruit_plant, params, generator, context ); 
    
  }

  }

}
  


void CanopyGenerator::walnut(const WalnutCanopyParameters &params, const vec3 &origin ){

  std::vector<uint> leaf_prototype = leafPrototype( params, generator, context );

  std::vector<uint> nut_prototype = nutPrototype( params, context );

  std::vector<uint> UUIDs;
  std::vector<uint> UUID_branch_plant;
  std::vector<std::vector<uint> > UUID_leaf_plant;
  std::vector<std::vector<std::vector<uint> > > UUID_fruit_plant;
  UUID_fruit_plant.resize(1);

  std::uniform_real_distribution<float> unif_distribution;

  //------ trunk -------//

  uint ID_trunk = addBranch( origin, make_vec3(0,0,1), params.trunk_radius, origin + make_vec3( unif_distribution(generator)*3*params.trunk_radius, unif_distribution(generator)*3*params.trunk_radius, params.trunk_height ),  make_vec3(0,0,0.5*params.trunk_height), 0.5*params.trunk_radius, 8, params, generator, context );

  UUID_trunk.push_back(context->getObjectPointer(ID_trunk)->getPrimitiveUUIDs());

  std::vector<vec3> pos_trunk = context->getTubeObjectPointer(ID_trunk)->getNodes();
  std::vector<float> rad_trunk = context->getTubeObjectPointer(ID_trunk)->getNodeRadii();

  branchRecursion( pos_trunk, rad_trunk, 1, 3, leaf_prototype, nut_prototype, UUID_branch_plant, UUID_leaf_plant, UUID_fruit_plant.front(), params, generator, context );


  context->deletePrimitive( leaf_prototype );
  context->deletePrimitive( nut_prototype );

  UUID_branch.push_back(UUID_branch_plant);
  UUID_leaf.push_back(UUID_leaf_plant);
  UUID_fruit.push_back(UUID_fruit_plant);

}
