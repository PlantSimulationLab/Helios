/** \file "WeberPennTree.cpp" Primary source file for Weber-Penn tree architecture model.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "WeberPennTree.h"
#ifndef _WIN32
#include <chrono>
#include <unistd.h>
#endif

using namespace helios;

WeberPennTree::WeberPennTree( helios::Context* __context ){

  context = __context;

  loadXML("plugins/weberpenntree/xml/WeberPennTreeLibrary.xml");

  branchLevels = 2; //default number of branching levels for which primitives should be generated

  trunk_segs = 30; //default number of radial triangular subdivisions of the trunk
  
  branch_segs = 6; //default number of radial triangular subdivisions of branches

  leaf_segs = make_int2(1,1); //default number of patch subdivisions of each leaf

  //seed the random number generator
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);

}

int WeberPennTree::selfTest( void ){

  std::cout << "Building trees in the default library..." << std::flush;

  float spacing = 5;

  std::vector<std::string> trees;
  trees.push_back("Almond");
  trees.push_back("Apple");
  trees.push_back("Avocado");
  trees.push_back("Lemon");
  trees.push_back("Olive");
  trees.push_back("Orange");
  trees.push_back("Peach");
  trees.push_back("Pistachio");
  trees.push_back("Walnut");

  for( int i=0; i<trees.size(); i++ ){

    Context context_1;

    WeberPennTree weberpenntree_1( &context_1 );

    weberpenntree_1.setBranchRecursionLevel(1);
    weberpenntree_1.setLeafSubdivisions( make_int2(3,3) );

    vec3 origin(i*spacing,0,0);
    
    weberpenntree_1.buildTree( trees.at(i).c_str(), origin, 0.75f );

  }

  std::cout << "done." << std::endl;
  return 0;
  
}

uint WeberPennTree::buildTree( const char* treename, helios::vec3 origin  ){
  return buildTree(treename,origin,1);
}

uint WeberPennTree::buildTree( const char* treename, helios::vec3 origin, float scale  ){

  if( trees_library.find(treename)==trees_library.end() ){
    std::cerr << "ERROR (WeberPennTree::buildTree): Tree " << treename << " does not exist in the tree library." << std::endl;
    exit(EXIT_FAILURE);
  }

  WeberPennTreeParameters parameters = trees_library.at(treename);

  uint TreeID = UUID_trunk.size();
  UUID_trunk.resize(TreeID+1);
  UUID_branch.resize(TreeID+1);
  UUID_leaf.resize(TreeID+1);

  //read the leaf mask
  std::string file = parameters.LeafFile;
  if( file.substr(file.find_last_of(".") + 1) != "png") {
    std::cerr << "ERROR (WeberPennTree::buildTree): Leaf image file for tree " << treename << " is not a PNG image." << std::endl;
    exit(EXIT_FAILURE);
  #ifndef _WIN32
  }else if( access( file.c_str(), F_OK ) == -1 ){
    std::cerr << "ERROR (WeberPennTree::buildTree): Leaf image file " << treename <<  " does not exist." << std::endl;
    exit(EXIT_FAILURE);
  #endif
  }

  uint ID_leaf_template = context->addTileObject( make_vec3(0,0,0), make_vec2(parameters.LeafScale*scale, parameters.LeafScale*parameters.LeafScaleX*scale), make_SphericalCoord(0,M_PI), leaf_segs, parameters.LeafFile.c_str() );
     
  uint base_nodes = 20;

  vec3 center = make_vec3(0,0,0);
 
  uint n;
  float z, theta, phi;
  
  std::vector<vec3> nodes;
  std::vector<float> radius;
  std::vector<RGBcolor> color;

  // --- Main trunk ---- //

  // Vectors to be passed to "addTube()"
  uint node_length;
  if( parameters.BaseSplits == 0 ){
    node_length = parameters.nCurveRes.at(0)+1;
  }else{
    node_length = base_nodes;
  }
  nodes.resize(node_length); //(x,y,z) coordinates of nodes along the branch
  radius.resize(node_length); //radius of the branch at each node location
  color.resize(node_length); //color of the branch at each node location

  // Length of trunk
  float length0 = (parameters.Scale + getVariation(parameters.ScaleV) )*(parameters.nLength[0] + getVariation(parameters.nLengthV[0]) )*scale;

  assert( length0>0 );

  // Length of trunk segments
  float dlength0;

  // Total number of stems along trunk
  uint stems = parameters.nBranches.at(1);
  
  // Number of stems per trunk segment
  uint stems_per_segment = round(float(stems)/float(parameters.nCurveRes.at(0)));

  //first (0) node of trunk @ ground
  nodes.at(0) = center;
  float flarez = parameters.Flare*(100.f-1.f)/100.f+1.f;
  radius.at(0) = length0*parameters.Ratio*parameters.ZScale*flarez;
  color.at(0) = parameters.WoodColor;

  assert( base_nodes>=2 );

  //Make base of trunk (region between ground and split or first branches)

  theta = 1e-5;
  phi = getVariation(2.f*M_PI);

  //Region above trunk base

  if( parameters.BaseSplits > 0 ){ //trunk splits at base

    dlength0 = length0*(1.f-parameters.BaseSplitSize)/float(parameters.nCurveRes.at(0));

    // Length of trunk base
    float base0;
    if( parameters.BaseSplits > 0 ){
      base0 = parameters.BaseSplitSize;
    }else{
      base0 = parameters.BaseSize;
    }

    for( uint i=1; i<base_nodes; i++ ){

      //nodes.at(i) = nodes.at(i-1) + rotatePoint( make_vec3( 0, 0, length0*(1-parameters.BaseSize)/parameters.nCurveRes.at(0) ), theta, phi );
      nodes.at(i) = nodes.at(i-1) + make_vec3( 0, 0, length0*base0/float(base_nodes) );

      float y = std::max( 0.f, 1.f-8.f*nodes.at(i).z/length0 );
      float flarez = parameters.Flare*(pow(100.f,y)-1.f)/100.f+1.f;
      radius.at(i) = length0*parameters.Ratio*(1-parameters.nTaper.at(0)*nodes.at(i).z/length0)*flarez;

      color.at(i) = parameters.WoodColor;
      
    }
    
    vec3 base_position = nodes.at(base_nodes-1);
    
    float offset_child = parameters.BaseSplitSize*length0;
    
    float phi_split = 0;//getVariation(2.f*M_PI);
    
    for( uint j=0; j<parameters.BaseSplits+1; j++ ){ //looping over clones (splits)

      float angle_split = (parameters.nSplitAngle.at(0)-getVariation(parameters.nSplitAngleV.at(0)))*M_PI/180.f;
      
      vec3 current_normal = make_vec3(0,0,1);

      helios::SphericalCoord child_rotation = make_SphericalCoord(angle_split,phi_split);
      
      recursiveBranch( parameters, 0, 0, base_position, current_normal, child_rotation, length0-offset_child, radius.at(base_nodes-1), offset_child, origin, scale, ID_leaf_template );
      
      //phi_split += (20+0.75*120*pow(getVariation(1),2))*M_PI/180.f;
      phi_split += 2.f*M_PI/float(parameters.BaseSplits+1);


    }
      
  }else{ //trunk does not split at base

    dlength0 = length0/float(parameters.nCurveRes.at(0));

    float phi_child = getVariation(2.f*M_PI);

    // Loop over trunk segments 1-0CurveRes
    for( uint i=1; i<parameters.nCurveRes.at(0); i++ ){

      //---- nodes ----//      

      nodes.at(i) = nodes.at(i-1) + rotatePoint( make_vec3( 0, 0, dlength0 ), theta, phi );

      vec3 current_normal = nodes.at(i)-nodes.at(i-1);
      current_normal.normalize();

      //---- radius ----//
      
      float unit_taper;
      if( parameters.nTaper.at(0)<1 ){
  	unit_taper = parameters.nTaper.at(0);
      }else if( parameters.nTaper.at(0)<2 ){
  	unit_taper = 2.f-parameters.nTaper.at(0);
      }else{
  	unit_taper = 0;
      }

      //fraction along trunk for lower (i-1) node
      float Zminus = float(i-1)/float(parameters.nCurveRes.at(0)-1);

      //fraction along trunk for upper (i) node
      float Zplus = float(i)/float(parameters.nCurveRes.at(0)-1);

      radius.at(i) = length0*parameters.Ratio*parameters.ZScale*(1.f-parameters.nTaper.at(0)*Zplus);
      assert( radius.at(i)>=0 );

      //---- color -----//
      
      color.at(i) = parameters.WoodColor;
      
      // --- Branch Recursion ---- //

      if( parameters.Levels>0 && Zplus>=parameters.BaseSize ){
      
	for( uint s=0; s<stems_per_segment; s++ ){
	
	  vec3 base_position = nodes.at(i-1) + (nodes.at(i)-nodes.at(i-1))*float(s+0.5)/float(stems_per_segment);
	  
	  float downangle = (parameters.nDownAngle.at(1)+getVariation(parameters.nDownAngleV.at(1)))*M_PI/180.f;
	  helios::SphericalCoord child_rotation = make_SphericalCoord(downangle,phi_child);
	  
	  float radius_parent = radius.at(i-1)+base_position.z/length0*(radius.at(i)-radius.at(i-1));
	  	  
	  //float offset_child = parameters.BaseSize*length0+(i-2)*dlength0 + dlength0*float(s)/float(stems_per_segment-1);
	  float base_nodes = floor(parameters.BaseSize*parameters.nCurveRes.at(0));
	  float offset_child = (Zminus-parameters.BaseSize)*length0+float(s+0.5)/float(stems_per_segment-1)*dlength0;

	  phi_child += (parameters.nRotate.at(1)+getVariation(parameters.nRotateV.at(1)))*M_PI/180.f;

	  if( offset_child>0 ){
	    recursiveBranch( parameters, 1, 0, base_position, current_normal, child_rotation, length0, radius_parent, offset_child, origin, scale, ID_leaf_template );
	  }
	  
	}
      }
      
      theta += (-parameters.nCurve.at(0)+getVariation(parameters.nCurveV.at(0)))/float(parameters.nCurveRes.at(0))*M_PI/180.f;
	
    }

  }

  //shift to origin
  for( uint i=0; i<nodes.size(); i++ ){
    nodes.at(i) = nodes.at(i) + origin;
  }

  //draw the trunk
  if( strcmp(parameters.WoodFile.c_str(),"none")==0 || parameters.WoodFile.size()==0 ){
    UUID_trunk.back() = context->addTube(trunk_segs,nodes,radius,color);
  }else{
    UUID_trunk.back() = context->addTube(trunk_segs,nodes,radius,parameters.WoodFile.c_str());
  }

  // printf("size of UUID_trunk = %d\n",UUID_trunk.back().size());
  // printf("trunk_segs = %d\n",trunk_segs);
  // printf("tube nodes = %d\n",radius.size());

  // for( int i=0; i<output_prim_data.size(); i++ ){
  //   if( output_prim_data.at(i).compare("branch_radius")==0 ){
  //     for( int p=0; p<UUID_trunk.back().size(); p++ ){
  // 	context->setPrimitiveData(UUID_trunk.back().at(p),"branch_radius",radius.at(c));
  //     }
  //   }else if( output_prim_data.at(i).compare("element_type")==0 ){
  //     context->setPrimitiveData(UUID_trunk.back(),"element_type","trunk");
  //   }
  // }

  //context->deletePrimitive( UUID_leaf_template );
  context->deleteObject( ID_leaf_template );

  uint flag=1;
  for( size_t i=0; i<UUID_leaf.at(TreeID).size(); i++ ){
    context->setPrimitiveData(UUID_leaf.at(TreeID).at(i),"twosided_flag",HELIOS_TYPE_UINT,1,&flag);
  }
  for( size_t i=0; i<UUID_branch.at(TreeID).size(); i++ ){
    context->setPrimitiveData(UUID_branch.at(TreeID).at(i),"twosided_flag",HELIOS_TYPE_UINT,1,&flag);
  }
  for( size_t i=0; i<UUID_trunk.at(TreeID).size(); i++ ){
    context->setPrimitiveData(UUID_trunk.at(TreeID).at(i),"twosided_flag",HELIOS_TYPE_UINT,1,&flag);
  }

  return TreeID;

}

void WeberPennTree::recursiveBranch( WeberPennTreeParameters parameters, uint n, uint seg_start, helios::vec3 base_position, helios::vec3 parent_normal, helios::SphericalCoord child_rotation, float length_parent, float radius_parent, float offset_child, helios::vec3 origin, float scale, const uint leaf_template ){

  if( n<parameters.Levels ){ //Branches

    if( n>=parameters.nCurveRes.size() ){ 
      n = parameters.Levels-1;
    }

    // Vectors to be passed to "addTube()"
    std::vector<vec3> nodes; //(x,y,z) coordinates of nodes along the branch
    std::vector<float> radius; //radius of the branch at each node location
    std::vector<RGBcolor> color; //color of the branch at each node location
    
    if( parameters.nSegSplits.at(n)==0 ){
      nodes.resize(parameters.nCurveRes.at(n)+1); 
      radius.resize(parameters.nCurveRes.at(n)+1); 
      color.resize(parameters.nCurveRes.at(n)+1);
    }else{
      nodes.resize(2); 
      radius.resize(2); 
      color.resize(2);
    }

    // Ratio of position along parent
    float ratio;
    if( n<=1 ){
      ratio = (length_parent*(1-parameters.BaseSize)-offset_child)/(length_parent*(1-parameters.BaseSize));
    }else{
      ratio = (length_parent-offset_child)/length_parent;
    }

					   
    if( ratio<0 ){
      ratio = 0;
    }else if(ratio>1){
      ratio = 1;
    }

    // Maximum length of the branch
    float length_childmax = parameters.nLength.at(n)+getVariation(parameters.nLengthV.at(n));
    
    if( length_childmax<0 ){
      length_childmax = parameters.nLength.at(n);
    }
    
    // Length of the branch
    float length_child;
    if( n==0 ){ //trunk splits
      length_child = length_parent*length_childmax;
    }else if( n<=1 ){ //first level of branches
      length_child = length_parent*length_childmax*ShapeRatio(parameters.Shape,ratio);
    }else{
      length_child = length_childmax*(length_parent-0.6*offset_child);
    }

    assert( length_child>=0 );

    // Length of each branch segment
    float dlength_child = length_child/float(parameters.nCurveRes.at(n));

    assert( dlength_child>=0 );

    // Total number of stems along branch
    uint stems;
    if( n==parameters.Levels-1 ){ //leaves
      //stems = 2.f*round(parameters.Leaves*ShapeRatio(4,ratio));
      //-- testing here --//
      stems = round(10.f*parameters.Leaves*length_child/parameters.Scale);
    }else{ //branches
      stems = round(parameters.nBranches.at(n+1)*(1-0.5*ratio));
      //stems = round(1.5*parameters.nBranches.at(n+1)*(1.f+ratio)*length_child/parameters.Scale);
    }
    
    // Number of stems per branch segment
    uint stems_per_segment = round(float(stems)/float(parameters.nCurveRes.at(n)));
    
    // First node
    if( n==0 ){
      radius.at(0) = 0.8*radius_parent;
    }else if( seg_start==0 ){
      radius.at(0) = radius_parent*pow(length_child/length_parent,parameters.RatioPower);
    }else{
      radius.at(0) = radius_parent*pow(length_child/length_parent,parameters.RatioPower)*(1-parameters.nTaper.at(n)*float(seg_start)/float(parameters.nCurveRes.at(n)));
    }
    nodes.at(0) = base_position;
    color.at(0) = parameters.WoodColor;

    SphericalCoord parent_rotation = cart2sphere(parent_normal);

    float phi = getVariation(2.f*M_PI);
    float phi_child = getVariation(2.f*M_PI);

    // Loop over branch segments 1-0CurveRes
    for( uint i=seg_start+1; i<parameters.nCurveRes.at(n)+1; i++ ){
      
      //---- radius ----//
      
      float unit_taper;
      if( parameters.nTaper.at(n)<1 ){
	unit_taper = parameters.nTaper.at(n);
      }else if( parameters.nTaper.at(n)<2 ){
	unit_taper = 2.f-parameters.nTaper.at(n);
      }else{
	unit_taper = 0;
      }
      
      float Z = float(i)/float(parameters.nCurveRes.at(n));

      //NOTE: STill not sure here if radius calculation is coming right when we have splits
      radius.at(i-seg_start) = radius.at(i-seg_start-1)*(1-parameters.nTaper.at(n)*Z);

      //---- nodes ----//
           
      vec3 child_normal = sphere2cart(make_SphericalCoord(parent_rotation.elevation-child_rotation.elevation,parent_rotation.azimuth));
      child_normal.normalize();

      child_normal = rotatePointAboutLine( child_normal, make_vec3(0,0,0), parent_normal, child_rotation.azimuth );
  
      nodes.at(i-seg_start) = nodes.at(i-seg_start-1) + child_normal*dlength_child;
       
      //---- color -----//
    
      color.at(i-seg_start) = parameters.WoodColor;
      
      // ---Branch recursion ---- //
      
      for( uint s=0; s<stems_per_segment; s++ ){

	child_rotation.elevation += (parameters.nCurve.at(n)+getVariation(parameters.nCurveV.at(n)))/float(parameters.nCurveRes.at(n))/float(stems_per_segment)*M_PI/180.f;

	float theta_child = child_rotation.elevation;
	  
      	float f;
      	if( stems_per_segment==1 ){
      	  f = 1;
      	}else{
      	  f= float(s+1)/float(stems_per_segment);
      	}

      	vec3 base_position = nodes.at(i-seg_start-1) + (nodes.at(i-seg_start)-nodes.at(i-seg_start-1))*f;

      	if( parameters.BaseSplits > 0 && n==0 && base_position.z<parameters.BaseSize*length_parent ){
      	  continue;
      	}
	
      	vec3 normal = nodes.at(i-seg_start)-nodes.at(i-seg_start-1);
      	normal.normalize();

      	float radius_p = radius.at(i-seg_start-1);
	
      	float offset_child = dlength_child*(i-seg_start-1)+dlength_child*float(s)/float(stems_per_segment);

      	float downangle = (parameters.nDownAngle.at(n+1)+getVariation(parameters.nDownAngleV.at(n+1)))*M_PI/180.f;
      	SphericalCoord rotation = make_SphericalCoord(downangle,phi_child);
	
      	recursiveBranch( parameters, n+1, 0, base_position, normal, rotation, length_child, radius_p, offset_child, origin, scale, leaf_template );

	if( n == parameters.Levels-1 ){//leaves
	  phi_child += (parameters.nRotate.at(n+1)+getVariation(parameters.nRotateV.at(n+1)))*M_PI/180.f;
	}else{//branches
	  phi_child += (parameters.nRotate.at(n+1)+getVariation(parameters.nRotateV.at(n+1)))*M_PI/180.f;
	}

      }
     
      // if nSegSplits>0, split into children and terminate current branch
      if( parameters.nSegSplits.at(n)>0 && i<parameters.nCurveRes.at(n) ){

	float angle_split = (parameters.nSplitAngle.at(n+1)-getVariation(parameters.nSplitAngleV.at(n+1)))*M_PI/180.f;
    
	vec3 base_position = nodes.back();
    
	float phi_split = 2.f*M_PI*getVariation(1);

	float radius_p = radius.at(i-seg_start);
    
	for( uint j=0; j<parameters.nSegSplits.at(n)+1; j++ ){ //looping over clones (splits)
      
	  vec3 normal = nodes.at(i-seg_start)-nodes.at(i-seg_start-1);
	  normal.normalize();

	  helios::SphericalCoord rotation = make_SphericalCoord(angle_split,phi_split);
      
	  recursiveBranch( parameters, n, i, base_position, normal, rotation, length_parent, radius_p, offset_child, origin, scale, leaf_template );
      
	  phi_split += 2.f*M_PI/float(parameters.nSegSplits.at(n)+1);
      
	}
	
	break;
      }
 
    }

    if( n<=branchLevels ){
      //shift to origin
      for( uint i=0; i<nodes.size(); i++ ){
	nodes.at(i) = nodes.at(i) + origin;
      }

      //add to context
      std::vector<uint> branches;
      if( strcmp(parameters.WoodFile.c_str(),"none")==0 ){
	branches = context->addTube(branch_segs,nodes,radius,color);
      }else{
	branches = context->addTube(branch_segs,nodes,radius,parameters.WoodFile.c_str());
      }
      UUID_branch.back().insert(UUID_branch.back().end(),branches.begin(),branches.end());
    }

  }else{ //leaves

    float phi = child_rotation.azimuth;
    
    vec3 position = base_position;

    SphericalCoord rotation = cart2sphere(parent_normal);

    //shift to origin
    position = position + origin;

    float downangle = (parameters.nDownAngle.at(n)+getVariation(parameters.nDownAngleV.at(n)))*M_PI/180.f;

    //calculate leaf angle from distibution
    if( parameters.leafAngleCDF.size()>0 ){
      float rt = context->randu();
      float dTheta = M_PI/float(parameters.leafAngleCDF.size());
      for( int i=0; i<parameters.leafAngleCDF.size(); i++ ){
	if( rt<parameters.leafAngleCDF.at(i) ){
	  float theta = (i+context->randu())*dTheta;
	  rotation = make_SphericalCoord(theta,context->randu()*2.f*M_PI);
	  break;
	}
      }
    }

    //add to Context
    uint ID = context->copyObject( leaf_template );
    std::vector<uint> UUIDs = context->getObjectPointer(ID)->getPrimitiveUUIDs();
    UUID_leaf.back().insert( UUID_leaf.back().end(), UUIDs.begin(), UUIDs.end() ); 
      
    //perform transformations
    if( parameters.leafAngleCDF.size()>0 ){
      context->getObjectPointer(ID)->rotate( rotation.elevation, "y" );
      context->getObjectPointer(ID)->rotate( rotation.azimuth, "z" );
      context->getObjectPointer(ID)->translate(position);
    }else{

      context->getObjectPointer(ID)->rotate( M_PI, "x" ); //flip leaf (this is so lighting looks right based on leaf normal)

      //rotate leaf so the tip is pointing in the same direction as the branch
      context->getObjectPointer(ID)->rotate( 0.5*M_PI - rotation.zenith, "y" );
      context->getObjectPointer(ID)->rotate( -0.5*M_PI - rotation.azimuth, "z" );

      vec3 lnorm = context->getTileObjectPointer(ID)->getNormal(); // current leaf normal vector
      vec3 pvec = cross( parent_normal, lnorm );

      //rotate leaf based on downangle
      if( downangle<0.f ){
	downangle = M_PI+downangle;
      }
      context->getObjectPointer(ID)->rotate( -downangle, pvec );

      //shift leaf perpendicular to the direction of the branch
      vec3 offset = -0.6*parameters.LeafScale*scale*fabs(sinf(downangle))*lnorm;
	
      context->getObjectPointer(ID)->translate( offset );

      //random azimuthal rotation about the branch
      context->getObjectPointer(ID)->rotate( phi, parent_normal );

      //translate to the position of the branch
      context->getObjectPointer(ID)->translate(position);
	
    }
      
  }

}

float WeberPennTree::ShapeRatio( uint shape, float ratio ){

  if( shape==0 ){ //conical
    return 0.2+0.8*ratio;
  }else if( shape==1 ){ //spherical
    return 0.2+0.8*sin(M_PI*ratio);
  }else if( shape==2 ){ //hemispherical
    return 0.2+0.8*sin(0.5*M_PI*ratio);
  }else if( shape==3 ){ //cylindrical
    return 1.f;
  }else if( shape==4 ){ //tapered cylinder
    return 0.5+0.5*ratio;
  }else if( shape==5 ){ //flame
    if( ratio<=0.7 ){
      return ratio/0.7f;
    }else{
      return (1.f-ratio)/0.3f;
    }
  }else if( shape==6 ){ //inverse conical
    return 1-0.8*ratio;
  }else if( shape==7 ){ //tend flame
    if( ratio<=0.7 ){
      return 0.5+0.5*ratio/0.7f;
    }else{
      return 0.5+0.5*(1-ratio)/0.3f;
    }
  }else if( shape==8 ){ //envelope
    return 0;
  }

  return 0;
  
}

float WeberPennTree::getVariation( float V ){

  std::uniform_real_distribution<float> unif_distribution;

  return -V + 2.f*unif_distribution(generator)*V;
  
}

std::vector<uint> WeberPennTree::getTrunkUUIDs( const uint TreeID ){

  if( TreeID >= UUID_trunk.size() ){
    std::cerr << "ERROR (WeberPennTree::getTrunkUUIDs): Tree ID " << TreeID << " does not exist." << std::endl;
    throw(1);
  }
  
  return UUID_trunk.at(TreeID);
}

std::vector<uint> WeberPennTree::getBranchUUIDs( const uint TreeID ){

  if( TreeID >= UUID_branch.size() ){
    std::cerr << "ERROR (WeberPennTree::getBranchUUIDs): Tree ID " << TreeID << " does not exist." << std::endl;
    throw(1);
  }
  
  return UUID_branch.at(TreeID);
}

std::vector<uint> WeberPennTree::getLeafUUIDs( const uint TreeID ){

  if( TreeID >= UUID_leaf.size() ){
    std::cerr << "ERROR (WeberPennTree::getLeafUUIDs): Tree ID " << TreeID << " does not exist." << std::endl;
    throw(1);
  }
  
  return UUID_leaf.at(TreeID);
}

std::vector<uint> WeberPennTree::getAllUUIDs( const uint TreeID ){

  if( TreeID >= UUID_leaf.size() ){
    std::cerr << "ERROR (WeberPennTree::getAllUUIDs): Tree ID " << TreeID << " does not exist." << std::endl;
    throw(1);
  }
  
  std::vector<uint> UUIDs;
  UUIDs.insert(UUIDs.end(),UUID_trunk.at(TreeID).begin(),UUID_trunk.at(TreeID).end());
  UUIDs.insert(UUIDs.end(),UUID_branch.at(TreeID).begin(),UUID_branch.at(TreeID).end());
  UUIDs.insert(UUIDs.end(),UUID_leaf.at(TreeID).begin(),UUID_leaf.at(TreeID).end());
  return UUIDs;
}

void WeberPennTree::setBranchRecursionLevel( const uint level ){
  branchLevels = level;
}

void WeberPennTree::setTrunkSegmentResolution( const uint segs ){
  if( segs<3 ){
    std::cout << "WARNING (WeberPennTree::setTrunkSegmentResolution): cannot set trunk segment resolution less than 3. Ignoring this command." << std::endl;
  }else{
    trunk_segs=segs;
  }
}

void WeberPennTree::setBranchSegmentResolution( const uint segs ){
  if( segs<3 ){
    std::cout << "WARNING (WeberPennTree::setBranchSegmentResolution): cannot set branch segment resolution less than 3. Ignoring this command." << std::endl;
  }else{
    branch_segs=segs;
  }
}

void WeberPennTree::setLeafSubdivisions( const helios::int2 segs ){
  if( segs.x<1 || segs.y<1 ){
    std::cout << "WARNING (WeberPennTree::setLeafSubdivisions): cannot set number of leaf subdivisions to less than 1. Ignoring this command." << std::endl;
  }else{
    leaf_segs=segs;
  }
}

WeberPennTreeParameters WeberPennTree::getTreeParameters( const char* treename ){

  if( trees_library.find(treename)==trees_library.end() ){
    std::cerr << "ERROR (WeberPennTree::getTreeParameters): Tree " << treename << " does not exist in the tree library." << std::endl;
    exit(EXIT_FAILURE);
  }

  return trees_library.at(treename);

}

void WeberPennTree::setTreeParameters( const char* treename, const WeberPennTreeParameters parameters ){

  if( trees_library.find(treename)==trees_library.end() ){
    std::cerr << "ERROR (WeberPennTree::setTreeParameters): Tree " << treename << " does not exist in the tree library." << std::endl;
    exit(EXIT_FAILURE);
  }

  trees_library.at(treename) = parameters;

}

void WeberPennTree::seedRandomGenerator( const uint seed ){
  generator.seed(seed);
}

void WeberPennTree::loadXML( const char* filename ){

  std::cout << "Loading Weber Penn Tree library from XML file: " << filename << "..." << std::flush;

  // Using "pugixml" parser.  See pugixml.org
  pugi::xml_document xmldoc;

  //load file
  pugi::xml_parse_result result = xmldoc.load_file(filename);

  //error checking
  if (!result){
    std::cout << "failed." << std::endl;
    std::cerr << "XML [" << filename << "] parsed with errors, attr value: [" << xmldoc.child("node").attribute("attr").value() << "]\n";
    std::cerr << "Error description: " << result.description() << "\n";
    std::cerr << "Error offset: " << result.offset << " (error at [..." << (filename + result.offset) << "]\n\n";
    exit(EXIT_FAILURE);
  }

  pugi::xml_node helios = xmldoc.child("helios");

  if( helios.empty() ){
    std::cout << "failed." << std::endl;
    std::cerr << "ERROR (loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags." << std::endl;
    exit(EXIT_FAILURE);
  }

  int tree_count = 0;
  for (pugi::xml_node p = helios.child("WeberPennTree"); p; p = p.next_sibling("WeberPennTree")){

    tree_count ++;

    WeberPennTreeParameters params;
    
    const char* label = p.attribute("label").value();
    params.label = label;

    // * Shape ID * //
    pugi::xml_node shape_node = p.child("Shape");
    
    //note: pugi loads xml data as a character.  need to separate it into 3 floats
    const char* shape_str = shape_node.child_value();
    if( strlen(shape_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Shape was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Shape = atoi(shape_str);
    }

    // * Base Size * //
    pugi::xml_node basesize_node = p.child("BaseSize");
    
    const char* basesize_str = basesize_node.child_value();
    if( strlen(basesize_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): BaseSize was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.BaseSize = atof(basesize_str);
    }

    // * Base Splits * //
    pugi::xml_node basesplits_node = p.child("BaseSplits");
    
    const char* basesplits_str = basesplits_node.child_value();
    if( strlen(basesplits_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): BaseSplits was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.BaseSplits = atoi(basesplits_str);
    }

    // * Base Split Size * //
    pugi::xml_node basesplitsize_node = p.child("BaseSplitSize");
    
    const char* basesplitsize_str = basesplitsize_node.child_value();
    if( strlen(basesplitsize_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): BaseSplitSize was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.BaseSplitSize = atof(basesplitsize_str);
    }

    // * Scale * //
    pugi::xml_node scale_node = p.child("Scale");
    
    const char* scale_str = scale_node.child_value();
    if( strlen(scale_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Scale was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Scale= atof(scale_str);
    }

    // * ScaleV * //
    pugi::xml_node scalev_node = p.child("ScaleV");
    
    const char* scalev_str = scalev_node.child_value();
    if( strlen(scalev_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): ScaleV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.ScaleV= atof(scalev_str);
    }

    // * ZScale * //
    pugi::xml_node zscale_node = p.child("ZScale");
    
    const char* zscale_str = zscale_node.child_value();
    if( strlen(zscale_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): ZScale was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.ZScale= atof(zscale_str);
    }

    // * ZScaleV * //
    pugi::xml_node zscalev_node = p.child("ZScaleV");
    
    const char* zscalev_str = zscalev_node.child_value();
    if( strlen(zscalev_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): ZScaleV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.ZScaleV= atof(zscalev_str);
    }

    // * Ratio * //
    pugi::xml_node ratio_node = p.child("Ratio");
    
    const char* ratio_str = ratio_node.child_value();
    if( strlen(ratio_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Ratio was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Ratio = atof(ratio_str);
    }

    // * RatioPower * //
    pugi::xml_node ratiopower_node = p.child("RatioPower");
    
    const char* ratiopower_str = ratiopower_node.child_value();
    if( strlen(ratiopower_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): RatioPower was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.RatioPower = atof(ratiopower_str);
    }

    // * Lobes * //
    pugi::xml_node lobes_node = p.child("Lobes");
    
    const char* lobes_str = lobes_node.child_value();
    if( strlen(lobes_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Lobes was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Lobes = atoi(lobes_str);
    }

    // * LobeDepth * //
    pugi::xml_node lobedepth_node = p.child("LobeDepth");
    
    const char* lobedepth_str = lobedepth_node.child_value();
    if( strlen(lobedepth_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): LobeDepth was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.LobeDepth = atof(lobedepth_str);
    }

    // * Flare * //
    pugi::xml_node flare_node = p.child("Flare");
    
    const char* flare_str = flare_node.child_value();
    if( strlen(flare_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Flare was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Flare = atof(flare_str);
    }

    // * Levels * //
    pugi::xml_node levels_node = p.child("Levels");
    
    const char* levels_str = levels_node.child_value();
    if( strlen(levels_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Levels was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Levels = atoi(levels_str);
    }

    int endLevel;
    if( params.Levels==4 ){
      endLevel=4;
    }else{
      endLevel=params.Levels+1;
    }

    
    // * nSegSplits * //
    pugi::xml_node nsegsplits_node = p.child("nSegSplits");

     const char* nsegsplits_str = nsegsplits_node.child_value();
    if( strlen(nsegsplits_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nSegSplits was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nSegSplits.resize(4);
    std::istringstream nsegsplits_stream(nsegsplits_str);
    for( int i=0; i<endLevel; i++ ){
      if( nsegsplits_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nSegSplits was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nsegsplits_stream >> params.nSegSplits.at(i);
      }
    }

    // * nSplitAngle * //
    pugi::xml_node nsplitangle_node = p.child("nSplitAngle");

     const char* nsplitangle_str = nsplitangle_node.child_value();
    if( strlen(nsplitangle_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nSplitAngle was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nSplitAngle.resize(4);
    std::istringstream nsplitangle_stream(nsplitangle_str);
    for( int i=0; i<endLevel; i++ ){
      if( nsplitangle_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nSplitAngle was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nsplitangle_stream >> params.nSplitAngle.at(i);
      }
    }

    // * nSplitAngleV * //
    pugi::xml_node nsplitanglev_node = p.child("nSplitAngleV");

     const char* nsplitanglev_str = nsplitanglev_node.child_value();
    if( strlen(nsplitanglev_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nSplitAngleV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nSplitAngleV.resize(4);
    std::istringstream nsplitanglev_stream(nsplitanglev_str);
    for( int i=0; i<endLevel; i++ ){
      if( nsplitanglev_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nSplitAngleV was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nsplitanglev_stream >> params.nSplitAngleV.at(i);
      }
    }

    // * nCurveRes * //
    pugi::xml_node ncurveres_node = p.child("nCurveRes");

     const char* ncurveres_str = ncurveres_node.child_value();
    if( strlen(ncurveres_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nCurveRes was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nCurveRes.resize(4);
    std::istringstream ncurveres_stream(ncurveres_str);
    for( int i=0; i<endLevel; i++ ){
      if( ncurveres_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nCurveRes was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ncurveres_stream >> params.nCurveRes.at(i);
      }
    }

    // * nCurve * //
    pugi::xml_node ncurve_node = p.child("nCurve");

     const char* ncurve_str = ncurve_node.child_value();
    if( strlen(ncurve_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nCurve was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nCurve.resize(4);
    std::istringstream ncurve_stream(ncurve_str);
    for( int i=0; i<endLevel; i++ ){
      if( ncurve_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nCurve was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ncurve_stream >> params.nCurve.at(i);
      }
    }

    // * nCurveV * //
    pugi::xml_node ncurvev_node = p.child("nCurveV");

     const char* ncurvev_str = ncurvev_node.child_value();
    if( strlen(ncurvev_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nCurveV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nCurveV.resize(4);
    std::istringstream ncurvev_stream(ncurvev_str);
    for( int i=0; i<endLevel; i++ ){
      if( ncurvev_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nCurveV was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ncurvev_stream >> params.nCurveV.at(i);
      }
    }

    // * nCurveBack * //
    pugi::xml_node ncurveback_node = p.child("nCurveBack");

     const char* ncurveback_str = ncurveback_node.child_value();
    if( strlen(ncurveback_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nCurveBack was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nCurveBack.resize(4);
    std::istringstream ncurveback_stream(ncurveback_str);
    for( int i=0; i<endLevel; i++ ){
      if( ncurveback_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nCurveBack was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ncurveback_stream >> params.nCurveBack.at(i);
      }
    }

    // * nLength * //
    pugi::xml_node nlength_node = p.child("nLength");

     const char* nlength_str = nlength_node.child_value();
    if( strlen(nlength_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nLength was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nLength.resize(4);
    std::istringstream nlength_stream(nlength_str);
    for( int i=0; i<endLevel; i++ ){
      if( nlength_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nLength was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nlength_stream >> params.nLength.at(i);
      }
    }

    // * nLengthV * //
    pugi::xml_node nlengthv_node = p.child("nLengthV");

     const char* nlengthv_str = nlengthv_node.child_value();
    if( strlen(nlengthv_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nLengthV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nLengthV.resize(4);
    std::istringstream nlengthv_stream(nlengthv_str);
    for( int i=0; i<endLevel; i++ ){
      if( nlengthv_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nLengthV was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nlengthv_stream >> params.nLengthV.at(i);
      }
    }

    // * nTaper * //
    pugi::xml_node ntaper_node = p.child("nTaper");

     const char* ntaper_str = ntaper_node.child_value();
    if( strlen(ntaper_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nTaper was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nTaper.resize(4);
    std::istringstream ntaper_stream(ntaper_str);
    for( int i=0; i<endLevel; i++ ){
      if( ntaper_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nTaper was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ntaper_stream >> params.nTaper.at(i);
      }
    }

    // * nDownAngle * //
    pugi::xml_node ndownangle_node = p.child("nDownAngle");

     const char* ndownangle_str = ndownangle_node.child_value();
    if( strlen(ndownangle_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nDownAngle was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nDownAngle.resize(4);
    std::istringstream ndownangle_stream(ndownangle_str);
    for( int i=0; i<endLevel; i++ ){
      if( ndownangle_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nDownAngle was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ndownangle_stream >> params.nDownAngle.at(i);
      }
    }

    // * nDownAngleV * //
    pugi::xml_node ndownanglev_node = p.child("nDownAngleV");

     const char* ndownanglev_str = ndownanglev_node.child_value();
    if( strlen(ndownanglev_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nDownAngleV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nDownAngleV.resize(4);
    std::istringstream ndownanglev_stream(ndownanglev_str);
    for( int i=0; i<endLevel; i++ ){
      if( ndownanglev_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nDownAngleV was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	ndownanglev_stream >> params.nDownAngleV.at(i);
      }
    }

    // * nRotate * //
    pugi::xml_node nrotate_node = p.child("nRotate");

     const char* nrotate_str = nrotate_node.child_value();
    if( strlen(nrotate_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nRotate was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nRotate.resize(4);
    std::istringstream nrotate_stream(nrotate_str);
    for( int i=0; i<endLevel; i++ ){
      if( nrotate_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nRotate was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nrotate_stream >> params.nRotate.at(i);
      }
    }

    // * nRotateV * //
    pugi::xml_node nrotatev_node = p.child("nRotateV");

     const char* nrotatev_str = nrotatev_node.child_value();
    if( strlen(nrotatev_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nRotateV was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nRotateV.resize(4);
    std::istringstream nrotatev_stream(nrotatev_str);
    for( int i=0; i<endLevel; i++ ){
      if( nrotatev_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nRotateV was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nrotatev_stream >> params.nRotateV.at(i);
      }
    }

    // * nBranches * //
    pugi::xml_node nbranches_node = p.child("nBranches");

     const char* nbranches_str = nbranches_node.child_value();
    if( strlen(nbranches_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): nBranches was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }
    

    params.nBranches.resize(4);
    std::istringstream nbranches_stream(nbranches_str);
    for( int i=0; i<endLevel; i++ ){
      if( nbranches_stream.peek()==EOF ){
	std::cerr << "ERROR (WeberPennTree::loadXML): nBranches was not given for level " << i << " of tree " << label << "." << std::endl;
	exit(EXIT_FAILURE);
      }else{
	nbranches_stream >> params.nBranches.at(i);
      }
    }

    // * Leaves * //
    pugi::xml_node leaves_node = p.child("Leaves");
    
    const char* leaves_str = leaves_node.child_value();
    if( strlen(leaves_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): Leaves was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.Leaves = atoi(leaves_str);
    }

    // * LeafFile * //
    pugi::xml_node leaffile_node = p.child("LeafFile");
    
    std::string leaffile_str = deblank(leaffile_node.child_value());
    if( leaffile_str.empty() ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): LeafFile was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.LeafFile = leaffile_str;
    }

    // * LeafScale * //
    pugi::xml_node leafscale_node = p.child("LeafScale");
    
    const char* leafscale_str = leafscale_node.child_value();
    if( strlen(leafscale_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): LeafScale was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.LeafScale = atof(leafscale_str);
    }

    // * LeafScaleX * //
    pugi::xml_node leafscalex_node = p.child("LeafScaleX");
    
    const char* leafscalex_str = leafscalex_node.child_value();
    if( strlen(leafscalex_str)==0 ){
      std::cout << "failed." << std::endl;
      std::cerr << "ERROR (WeberPennTree::loadXML): LeafScaleX was not given for tree " << label << "." << std::endl;
      exit(EXIT_FAILURE);
    }else{
      params.LeafScaleX = atof(leafscalex_str);
    }

    // * WoodColor * //
    pugi::xml_node woodcolor_node = p.child("WoodColor");
    
    const char* woodcolor_str = woodcolor_node.child_value();
    if( strlen(woodcolor_str)==0 ){
      params.WoodColor = make_RGBcolor(0.4,0.2,0.05);
    }else{
      RGBAcolor colort = string2RGBcolor(woodcolor_str);
      params.WoodColor = make_RGBcolor(colort.r,colort.g,colort.b);
    }

    // * WoodFile * //
    pugi::xml_node woodfile_node = p.child("WoodFile");
    
    std::string woodfile_str = deblank(woodfile_node.child_value());
    if( woodfile_str.empty() ){
      params.WoodFile = "none";
    }else{
      params.WoodFile = woodfile_str;
    }

    // * leafAngleDist * //
    pugi::xml_node gL_node = p.child("LeafAngleDist");

    const char* gL_string = gL_node.child_value();
    std::istringstream gL_stream(gL_string);
    std::vector<float> leafAngleDist;
    float tmp;
    while( gL_stream >> tmp ){
      leafAngleDist.push_back(tmp);
    }
    //calculate leaf angle CDF
    if( leafAngleDist.size()>0 ){
      float dTheta = M_PI/float(leafAngleDist.size());
      params.leafAngleCDF.resize( leafAngleDist.size() );
      float tsum = 0;
      for( int i=0; i<leafAngleDist.size(); i++ ){
	tsum += leafAngleDist.at(i)*dTheta;
	params.leafAngleCDF.at(i) = tsum;
      }
      if( fabs(tsum-1.f)>0.001 ){
	std::cout << "WARNING (WeberPennTree::loadXML): Leaf angle distribution for tree " << label << " does not sum to 1. Assuming isotropic." << std::endl;
	std::cout << params.leafAngleCDF.back() << std::endl;
	params.leafAngleCDF.resize(0);
      }else{
	params.leafAngleCDF.back() = 1.f;
      }
    }
  
    //store the parameters
    
    trees_library[label] = params;

  }

  if( tree_count==0 ){
    std::cout << "failed." << std::endl;
    std::cerr << "ERROR (WeberPennTree::loadXML): XML file " << filename << " did not contain any tree definitions." << std::endl;
  }else{
    std::cout << "done." << std::endl;
    std::cout << "Loaded " << tree_count << " tree definition(s)." << std::endl; 
  }
    

}

void WeberPennTree::optionalOutputPrimitiveData( const char* label ){

  if( strcmp(label,"element_type")==0 || strcmp(label,"branch_radius")==0 ){
    output_prim_data.push_back( label );
  }else{
    std::cout << "WARNING (WeberPennTree::optionalOutputPrimitiveData): unknown optional output primitive data " << label << std::endl;
  }
  
}
