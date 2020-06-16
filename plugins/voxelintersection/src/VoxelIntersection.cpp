/** \file "VoxelIntersection.cpp" Primary source file for Primitive-Voxel Intersection plug-in.
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

#include "VoxelIntersection.h"

using namespace helios;

VoxelIntersection::VoxelIntersection( helios::Context* __context ){

  context = __context;

  printmessages=true;

}

int VoxelIntersection::selfTest( void ){

  if( printmessages ){
    std::cout << "Running self-test for VoxelIntersection plug-in..." << std::flush;
  }
    
  Context context_test;

  vec3 vorigin = make_vec3(1,2,3);
  vec3 vsize = make_vec3(3,2,1);
  uint UUID_v = context_test.addVoxel( vorigin, vsize );

  helios::int3 Ninside = helios::make_int3(5,5,5);

  vec2 psize = make_vec2(0.1,0.1);

  std::vector<uint> UUID_inside, UUID_outside;

  for( int k=0; k<Ninside.z; k++ ){
    for( int j=0; j<Ninside.y; j++ ){
      for( int i=0; i<Ninside.x; i++ ){

	vec3 position = vorigin + make_vec3( -0.5*vsize.x+2*psize.x+i*(vsize.x-4*psize.x)/float(Ninside.x), -0.5*vsize.y+2*psize.y+j*(vsize.y-4*psize.y)/float(Ninside.y), -0.5*vsize.z+2*psize.x+k*(vsize.z-4*psize.x)/float(Ninside.z) );

	UUID_inside.push_back( context_test.addPatch( position, psize ) );
	
      }
    }
  }

  UUID_outside.push_back( context_test.addPatch( vorigin + 5*vsize, psize ) );
  UUID_outside.push_back( context_test.addPatch( vorigin - 5*vsize, psize ) );
  UUID_outside.push_back( context_test.addPatch( vorigin + 10*vsize, psize ) );
  UUID_outside.push_back( context_test.addPatch( vorigin - 10*vsize, psize ) );

  VoxelIntersection vintersection(&context_test);

  vintersection.calculatePrimitiveVoxelIntersection();

  std::vector<uint> UUIDs;

  context_test.getPrimitiveData( UUID_v, "inside_UUIDs", UUIDs );

  bool test_pass = true;
  for( int k=0; k<UUIDs.size(); k++ ){

    bool flag = false;
    for( int i=0; i<UUID_inside.size(); i++ ){
      if( UUIDs.at(k)==UUID_inside.at(i) ){
	flag = true;
	break;
      }
    }
    if( !flag ){
      test_pass = false;
      break;
    }

    for( int i=0; i<UUID_outside.size(); i++ ){
      if( UUIDs.at(k)==UUID_outside.at(i) ){
	test_pass = false;
	break;
      }
    }
    
  }

  for( int k=0; k<UUID_inside.size(); k++ ){

    bool flag = false;
    for( int i=0; i<UUIDs.size(); i++ ){
      if( UUIDs.at(i)==UUID_inside.at(k) ){
	flag = true;
	break;
      }
    }
    if( !flag ){
      test_pass = false;
      break;
    }
    
  }

  if( UUIDs.size()!=UUID_inside.size() ){
    test_pass = false;
  }

  if( test_pass ){
    if( printmessages ){
      std::cout << "passed." << std::endl;
    }
    return 0;
  }else{
    if( printmessages ){
      std::cout << "failed." << std::endl;
    }
    return 1;
  }
  
}
