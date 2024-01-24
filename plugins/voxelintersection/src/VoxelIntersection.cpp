/** \file "VoxelIntersection.cpp" Primary source file for Primitive-Voxel Intersection plug-in.

    Copyright (C) 2016-2024 Brian Bailey, Eric Kent

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
    std::cout << "Running self-test for VoxelIntersection plug-in..." << std::endl;
  }

  int error_count = 0;

  //////////////////////////////////////////////////////////////////////////////////////
  //------ single patch entirely on one side of voxel face should not be sliced ------//
  Context context_1;
  if( printmessages ){
    std::cout << "Running Test 1: patch with no slices..." << std::flush;
  }
  VoxelIntersection vi_1(&context_1);
  vi_1.disableMessages();
  
  //create single patch
  vec3 center_1 = make_vec3(0.0,0.0,0.0);
  vec2 size_1 = make_vec2(1,1);
  SphericalCoord rotation_1 = make_SphericalCoord(0,0);
  uint test_patch_1 = context_1.addPatch( center_1, size_1, rotation_1);
  
  std::vector<uint> allUUIDs_1 = context_1.getAllUUIDs();
  std::vector<uint> s1_1 = vi_1.slicePrimitivesUsingGrid(allUUIDs_1, make_vec3(0.0,0,0), make_vec3(1,1,1), make_int3(1,1,1));
  if(s1_1.size() == 1)
    {
      std::cout << "passed" << std::endl;
    }else{
    error_count ++;
    std::cerr << "failed: single patch inside voxel was sliced when it shouldn't have been" << std::endl;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  //------ single patch partially on one side of voxel face should be sliced to create 4 triangles ------//
  if( printmessages ){
    std::cout << "Running Test 2: patch with one slice..." << std::flush;
  }
    
  Context context_2;
    
  VoxelIntersection vi_2(&context_2);
  vi_2.disableMessages();
    
  //create single patch
  vec3 center_2 = make_vec3(0.0,0.0,0.0);
  vec2 size_2 = make_vec2(1,1);
  SphericalCoord rotation_2 = make_SphericalCoord(0,0);
  uint test_patch_2 = context_2.addPatch( center_2, size_2, rotation_2);
  
  std::vector<uint> allUUIDs_2 = context_2.getAllUUIDs();
    
  std::vector<uint> s1_2 = vi_2.slicePrimitivesUsingGrid(allUUIDs_2, make_vec3(0.5,0,0), make_vec3(1,1,1), make_int3(1,1,1));
  
  if(s1_2.size() == 4)
    {
      std::cout << "passed" << std::endl;
    }else{
    error_count ++;
    std::cerr << "failed: single patch that should be sliced into four triangles was sliced into " << s1_2.size() << std::endl;
  }
    
    
  //////////////////////////////////////////////////////////////////////////////////////
  //------ single patch intersects 2 voxel faces ------//
  if( printmessages ){
    std::cout << "Running Test 3: patch with 2 slices..." << std::flush;
  }
    
  Context context_3;
  
  VoxelIntersection vi_3(&context_3);
  vi_3.disableMessages();
  
  //create single patch
  vec3 center_3 = make_vec3(0.0,0.0,0.0);
  vec2 size_3 = make_vec2(1,1);
  SphericalCoord rotation_3 = make_SphericalCoord(0,0);
  uint test_patch_3 = context_3.addPatch( center_3, size_3, rotation_3);
  
  std::vector<uint> allUUIDs_3 = context_3.getAllUUIDs();
  
  std::vector<uint> s1_3 = vi_3.slicePrimitivesUsingGrid(allUUIDs_3, make_vec3(0.5,0.5,0), make_vec3(1,1,1), make_int3(1,1,1));
  
  if(s1_3.size() == 12)
    {
      std::cout << "passed" << std::endl;
    }else{
    error_count ++;
    std::cerr << "failed: single patch that should be sliced into 12 triangles was sliced into " << s1_3.size() << std::endl;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  //------ single patch intersects 3 voxel faces, no patch vertices inside the voxel ------//
  if( printmessages ){
    std::cout << "Running Test 4: patch with 3 slices and no vertices inside voxel..." << std::flush;
  }
  
  Context context_4;
  
  VoxelIntersection vi_4(&context_4);
  vi_4.disableMessages();
  
  //create single patch
  vec3 center_4 = make_vec3(0.05,0.05,0.05);
  vec2 size_4 = make_vec2(1,1);
  SphericalCoord rotation_4 = make_SphericalCoord(M_PI*0.25,M_PI*0.25);
  uint test_patch_4 = context_4.addPatch( center_4, size_4, rotation_4);
  
  helios::vec3 grid_center =  make_vec3(0.5,0.5,0.5);
  helios::vec3 grid_size = make_vec3(1,1,1);
  helios::int3 grid_divisions = make_int3(1,1,1);
  
  std::vector<uint> allUUIDs_4 = context_4.getAllUUIDs();
  std::vector<uint> s1_4 = vi_4.slicePrimitivesUsingGrid(allUUIDs_4, grid_center, grid_size, grid_divisions);
  
  int count = 0;
  for(uint i=0;i<s1_4.size();i++)
    {
      int cell_ID;
      context_4.getPrimitiveData(s1_4.at(i), "cell_ID", cell_ID);
      if(cell_ID >= 0)
        {
	  count++;
        }
    }
  
  if(count == 1)
    {
      std::cout << "passed" << std::endl;
    }else{
    error_count ++;
    std::cerr << "failed: a single triangle should have been classified as inside the voxel but " << count << " were inside."  << std::endl;
  }
  
  //////////////////////////////////////////////////////////////////////////////////////
  //------ many non-textured patches, check total area is consistent ------//
  if( printmessages ){
    std::cout << "Running Test 5: areas after slicing..." << std::flush;
  }
  Context context_5;
  VoxelIntersection vi_5(&context_5);
  vi_5.disableMessages();
  
  for(uint i=0;i<100;i++)
    {
      SphericalCoord rotation_5 = make_SphericalCoord(M_PI*randu(),M_PI*randu());
      context_5.addPatch( make_vec3(randu(),randu(),randu()), make_vec2(0.05, 0.05), rotation_5);
    }
  
  
  helios::vec3 grid_center_5 =  make_vec3(0.5,0.5,0.5);
  helios::vec3 grid_size_5 = make_vec3(1,1,1);
  helios::int3 grid_divisions_5 = make_int3(4,4,4);
  
  std::vector<uint> allUUIDs_5 = context_5.getAllUUIDs();
  
  float total_area_before_5 = 0;
  for(uint i=0;i<allUUIDs_5.size();i++)
    {
      total_area_before_5 = total_area_before_5 + context_5.getPrimitiveArea(allUUIDs_5.at(i));
    }
  
  std::vector<uint> s1_5 = vi_5.slicePrimitivesUsingGrid(allUUIDs_5, grid_center_5, grid_size_5, grid_divisions_5);
  std::vector<std::vector<uint>> cell_primitives_5 = vi_5.getCellPrimitives();
  
  float total_area_5 = 0;
  uint total_n_prim_5 = 0;
  for(int i=0;i<cell_primitives_5.size();i++)
    {
      float area_5=0;
      for(uint j=0;j<cell_primitives_5.at(i).size();j++)
        {
	  area_5 = area_5 + context_5.getPrimitiveArea(cell_primitives_5.at(i).at(j));
        }
      total_area_5 = total_area_5 + area_5;
    }
  float pdiff_5 = (total_area_5 - total_area_before_5)/ total_area_before_5 * 100;
  
  if(pdiff_5 < 1.0)
    {
      std::cout << "passed: area before and after slicing within 1% (" << pdiff_5 << "%)" << std::endl;
    }else{
    error_count ++;
    std::cerr << "failed: area before and after slicing different by" << pdiff_5 << "%"  << std::endl;
  }
      
  //////////////////////////////////////////////////////////////////////////////////////
  //------ many textured patches, check total area is consistent ------//
  if( printmessages ){
    std::cout << "Running Test 6: textured areas after slicing..." << std::flush;
  }
  Context context_6;
  VoxelIntersection vi_6(&context_6);
  vi_6.disableMessages();
  
  for(uint i=0;i<100;i++)
    {
      SphericalCoord rotation_6 = make_SphericalCoord(M_PI*randu(),M_PI*randu());
      context_6.addPatch( make_vec3(randu(),randu(),randu()), make_vec2(0.05, 0.05), rotation_6, "../../../plugins/visualizer/textures/GrapeLeaf.png", make_vec2(0.5, 0.5), make_vec2(1,1) );
    }
  
  helios::vec3 grid_center_6 =  make_vec3(0.5,0.5,0.5);
  helios::vec3 grid_size_6 = make_vec3(1,1,1);
  helios::int3 grid_divisions_6 = make_int3(4,4,4);
  
  std::vector<uint> allUUIDs_6 = context_6.getAllUUIDs();
    
  float total_area_before_6 = 0;
  for(uint i=0;i<allUUIDs_6.size();i++)
    {
      total_area_before_6 = total_area_before_6 + context_6.getPrimitiveArea(allUUIDs_6.at(i));
    }
  
  std::vector<uint> s1_6 = vi_6.slicePrimitivesUsingGrid(allUUIDs_6, grid_center_6, grid_size_6, grid_divisions_6);
  std::vector<std::vector<uint>> cell_primitives_6 = vi_6.getCellPrimitives();
  
  float total_area_6 = 0;
  uint total_n_prim_6 = 0;
  for(int i=0;i<cell_primitives_6.size();i++)
    {
      float area_6=0;
      for(uint j=0;j<cell_primitives_6.at(i).size();j++)
        {
	  area_6 = area_6 + context_6.getPrimitiveArea(cell_primitives_6.at(i).at(j));
        }
      total_area_6 = total_area_6 + area_6;
    }
  float pdiff_6 = (total_area_6 - total_area_before_6)/ total_area_before_6 * 100;
  
  if(pdiff_6 < 1.0)
    {
      std::cout << "passed: area before and after slicing within 1% (" << pdiff_6 << "%)" << std::endl;
    }else{
    error_count ++;
    std::cerr << "failed: area before and after slicing different by" << pdiff_6 << "%"  << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  //------ many non-textured patches, check total area is consistent ------//
  if( printmessages ){
      std::cout << "Running Test 7: cropping non-textured primitives..." << std::flush;
  }
  Context context_7;
  VoxelIntersection vi_7(&context_7);
  vi_7.disableMessages();
  
  for(uint i=0;i<100;i++)
  {
      SphericalCoord rotation_7 = make_SphericalCoord(M_PI*randu(),M_PI*randu());
      context_7.addPatch( make_vec3(randu(),randu(),randu()), make_vec2(0.05, 0.05), rotation_7);
  }
  
  helios::vec3 grid_center_7 =  make_vec3(0.5,0.5,0.5);
  helios::vec3 grid_size_7 = make_vec3(1,1,1);
  helios::int3 grid_divisions_7 = make_int3(4,4,4);
  
  std::vector<uint> allUUIDs_7 = context_7.getAllUUIDs();
  
  std::vector<uint> s1_7 = vi_7.slicePrimitivesUsingGrid(allUUIDs_7, grid_center_7, grid_size_7, grid_divisions_7);
  std::vector<std::vector<uint>> cell_primitives_7 = vi_7.getCellPrimitives();
  
  
  uint UUIDs_in_cells_7 = 0;
  for(uint i=0;i<s1_7.size();i++)
  {
      int cell_ID_7;
      context_7.getPrimitiveData(s1_7.at(i), "cell_ID", cell_ID_7);
      if(cell_ID_7 != -1){UUIDs_in_cells_7 ++;}
  }
  
  
  context_7.cropDomain(make_vec2(grid_center_7.x - grid_size_7.x*0.5 , grid_center_7.x + grid_size_7.x*0.5 ),make_vec2(grid_center_7.y - grid_size_7.y*0.5 , grid_center_7.y + grid_size_7.y*0.5 ),make_vec2(grid_center_7.z - grid_size_7.z*0.5 , grid_center_7.z + grid_size_7.z*0.5 ));
  
  
  std::vector<uint> UUIDs_after_crop_7 = context_7.getAllUUIDs();
  
  if( UUIDs_after_crop_7.size() == UUIDs_in_cells_7)
  {
      std::cout << "passed: number of primitives after cropping matches number in grid cells" << std::endl;
  }else{
      error_count ++;
      std::cerr << "failed: number of primitives after cropping doesn't match number in grid cells: " << UUIDs_after_crop_7.size() << " vs. " << UUIDs_in_cells_7 << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  //------ many textured patches, check number of primitives in cells matches number after cropping ------//
  if( printmessages ){
      std::cout << "Running Test 8: cropping textured primitives..." << std::flush;
  }
  Context context_8;
  VoxelIntersection vi_8(&context_8);
  vi_8.disableMessages();
  for(uint i=0;i<100;i++)
  {
      SphericalCoord rotation_8 = make_SphericalCoord(M_PI*randu(),M_PI*randu());
      context_8.addPatch( make_vec3(randu(),randu(),randu()), make_vec2(0.05, 0.05), rotation_8, "../../../plugins/visualizer/textures/GrapeLeaf.png", make_vec2(0.5, 0.5), make_vec2(1,1) );
      
  }
  
  helios::vec3 grid_center_8 =  make_vec3(0.5,0.5,0.5);
  helios::vec3 grid_size_8 = make_vec3(1,1,1);
  helios::int3 grid_divisions_8 = make_int3(4,4,4);
  
  std::vector<uint> allUUIDs_8 = context_8.getAllUUIDs();
  
  std::vector<uint> s1_8 = vi_8.slicePrimitivesUsingGrid(allUUIDs_8, grid_center_8, grid_size_8, grid_divisions_8);
  std::vector<std::vector<uint>> cell_primitives_8 = vi_8.getCellPrimitives();
  
  
  uint UUIDs_in_cells_8 = 0;
  for(uint i=0;i<s1_8.size();i++)
  {
      int cell_ID_8;
      context_8.getPrimitiveData(s1_8.at(i), "cell_ID", cell_ID_8);
      if(cell_ID_8 != -1){UUIDs_in_cells_8 ++;}
  }
  
  context_8.cropDomain(make_vec2(grid_center_8.x - grid_size_8.x*0.5 , grid_center_8.x + grid_size_8.x*0.5 ),make_vec2(grid_center_8.y - grid_size_8.y*0.5 , grid_center_8.y + grid_size_8.y*0.5 ),make_vec2(grid_center_8.z - grid_size_8.z*0.5 , grid_center_8.z + grid_size_8.z*0.5 ));
  
  std::vector<uint> UUIDs_after_crop_8 = context_8.getAllUUIDs();
  
  if( UUIDs_after_crop_8.size() == UUIDs_in_cells_8)
  {
      std::cout << "passed: number of primitives after cropping matches number in grid cells" << std::endl;
  }else{
      error_count ++;
      std::cerr << "failed: number of primitives after cropping doesn't match number in grid cells: " << UUIDs_after_crop_8.size() << " vs. " << UUIDs_in_cells_8 << std::endl;
  }

  //////////////////////////////////////////////////////////////////////////////////////
  //------ Voxel-patch intersection test ------//
  if( printmessages ){
    std::cout << "Running Test 9: intersection of voxels and patches..." << std::flush;
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
  vintersection.disableMessages();

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
  }else{
    if( printmessages ){
      std::cout << "failed." << std::endl;
    }
    error_count ++;
  }
  
  
  
  if( error_count==0 ){
    std::cout << " all tests passed." << std::endl;
    return 0;
  }else{
    return 1;
  }
  
  
}

// find the intersection point of two lines 
helios::vec3 VoxelIntersection::linesIntersection(helios::vec3 line1_point, helios::vec3 line1_direction, helios::vec3 line2_point, helios::vec3 line2_direction){
    
    helios::vec3 g = line2_point-line1_point;
    helios::vec3 h = cross(line2_direction,g);
    helios::vec3 k = cross(line2_direction,line1_direction);
    helios::vec3 rht;
    helios::vec3 out;
    
    float h_mag = sqrt(pow(h.x,2) + pow(h.y,2) + pow(h.z,2) );
    float k_mag = sqrt(pow(k.x,2) + pow(k.y,2) + pow(k.z,2) );
    
    //in the same direction
    if( ((h.x >= 0 && k.x >=0) || (h.x < 0 && k.x < 0)) && ((h.y >= 0 && k.y >=0) || (h.y < 0 && k.y < 0)) && ((h.z >= 0 && k.z >=0) || (h.z < 0 && k.z < 0)) ){
        rht = (h_mag/k_mag)*line1_direction;
        return line1_point + rht;
    }else{ // different direction
        rht = (h_mag/k_mag)*line1_direction;
        return line1_point - rht ;
    }
}

std::vector<uint> VoxelIntersection::slicePrimitive(uint UUID, std::vector<helios::vec3> voxel_face_vertices)
{
    
    //vector of UUIDs that will be output
    std::vector<uint> resulting_UUIDs;
    
    if(voxel_face_vertices.size() < 3)
    {
        std::cerr << "ERROR (slicePrimitive): voxel_face_verticies must contain at least three points" << std::endl;
        throw(1);
    }
    
    helios::vec3 face_normal = cross(voxel_face_vertices.at(1) - voxel_face_vertices.at(0), voxel_face_vertices.at(2) - voxel_face_vertices.at(1));
    face_normal.normalize();
    
    std::vector<helios::vec3> primitive_vertices = context->getPrimitiveVertices(UUID);
    helios::vec3 primitive_normal = context->getPrimitiveNormal(UUID);
    primitive_normal.normalize();
    
    helios::RGBAcolor primitive_color = context->getPrimitiveColorRGBA(UUID);
    
    std::string texa;
    const char * tex;
    texa =  context->getPrimitiveTextureFile(UUID);
    tex = texa.c_str();
    bool primitiveHasTexture = !texa.empty();
    
    //get the area of the original primitive for comparison with the area of the sliced primitives later
    float original_area = context->getPrimitiveArea(UUID);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find the equation of the line where the planes of the patch and voxel face intersect
    
    // direction of the plane intersection line
    helios::vec3 direction_vector = cross(face_normal, primitive_normal);
    
    // find a point on the plane intersection line
    // based on https://vicrucann.github.io/tutorials/3d-geometry-algorithms/
    helios::vec3 a = helios::make_vec3(fabs(direction_vector.x),  fabs(direction_vector.y), fabs(direction_vector.z));
    uint maxc;
    if(a.x > a.y)
    {
        if(a.x > a.z){
            maxc = 1;
        }else{
            maxc = 3;
        }
    }else{
        if(a.y > a.z)
        {
            maxc=2;
        }else{
            maxc = 3;
        }
    }
    
    helios::vec3 d1a = helios::make_vec3(-1*face_normal.x*voxel_face_vertices.at(0).x, -1*face_normal.y*voxel_face_vertices.at(0).y, -1*face_normal.z*voxel_face_vertices.at(0).z);
    helios::vec3 d2a = helios::make_vec3(-1*primitive_normal.x*primitive_vertices.at(1).x, -1*primitive_normal.y*primitive_vertices.at(1).y, -1*primitive_normal.z*primitive_vertices.at(1).z);
    
    float d1 = d1a.x + d1a.y + d1a.z;
    float d2 = d2a.x + d2a.y + d2a.z;
    
    float xi;
    float yi;
    float zi;
    
    if(maxc == 1)
    {
        xi = 0;
        yi  = (d2*face_normal.z - d1*primitive_normal.z) /  direction_vector.x;
        zi = (d1*primitive_normal.y - d2*face_normal.y) /  direction_vector.x;
    }else if(maxc == 2){
        xi = (d1*primitive_normal.z - d2*face_normal.z) /  direction_vector.y;
        yi = 0;
        zi = (d2*face_normal.x - d1*primitive_normal.x) /  direction_vector.y;
    }else if(maxc == 3){
        xi = (d2*face_normal.y - d1*primitive_normal.y) /  direction_vector.z;
        yi = (d1*primitive_normal.x - d2*face_normal.x) /  direction_vector.z;
        zi = 0 ;
    }
    
    helios::vec3  ipoint = make_vec3(xi, yi, zi);
    
    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // get points of intersection between each edge of the patch and the patch-voxel intersection line
    
    // vector for points of intersection between edge line and intersection line
    std::vector<helios::vec3> possible_points;
    // vector for points that actually touch the patch
    std::vector<helios::vec3> slice_points;
    std::vector<uint> slice_points_edge_ID;
    uint vertex_index; // index for cases where one slice point is on a vertex (used for patch cases only)
    
    helios::vec3 vi0;
    helios::vec3 vi1;
    
    // go through the different edges of the patch and calculate intersection points with line along edge of patch and intersection line
    if(primitive_vertices.size() == 4)
    {
        possible_points.resize(4);
        possible_points.at(0) = linesIntersection(primitive_vertices.at(1), primitive_vertices.at(1) - primitive_vertices.at(0), ipoint, direction_vector);
        possible_points.at(1) = linesIntersection(primitive_vertices.at(2), primitive_vertices.at(2) - primitive_vertices.at(1), ipoint, direction_vector);
        possible_points.at(2) = linesIntersection(primitive_vertices.at(3), primitive_vertices.at(3) - primitive_vertices.at(2), ipoint, direction_vector);
        possible_points.at(3) = linesIntersection(primitive_vertices.at(0), primitive_vertices.at(0) - primitive_vertices.at(3), ipoint, direction_vector);
        
        for(uint i=0;i<4;i++)
        {
            if(i == 0)
            {
                vi1 = primitive_vertices.at(1);
                vi0 = primitive_vertices.at(0);
            }else if(i == 1){
                vi1 = primitive_vertices.at(2);
                vi0 = primitive_vertices.at(1);
            }else if(i == 2){
                vi1 = primitive_vertices.at(3);
                vi0 = primitive_vertices.at(2);
            }else if(i == 3){
                vi1 = primitive_vertices.at(0);
                vi0 = primitive_vertices.at(3);
            }
            
            bool test_x = ( (possible_points.at(i).x >= vi1.x && possible_points.at(i).x <= vi0.x) || (possible_points.at(i).x >= vi0.x && possible_points.at(i).x <= vi1.x) );
            bool test_y = ( (possible_points.at(i).y >= vi1.y && possible_points.at(i).y <= vi0.y) || (possible_points.at(i).y >= vi0.y && possible_points.at(i).y <= vi1.y) );
            bool test_z = ( (possible_points.at(i).z >= vi1.z && possible_points.at(i).z <= vi0.z) || (possible_points.at(i).z >= vi0.z && possible_points.at(i).z <= vi1.z) );
            
            if( test_x && test_y && test_z){
                slice_points.push_back(possible_points.at(i));
                slice_points_edge_ID.push_back(i);
            }
        }
        
    }else if(primitive_vertices.size() == 3){
        
        possible_points.resize(3);
        possible_points.at(0) = linesIntersection(primitive_vertices.at(1), primitive_vertices.at(1) - primitive_vertices.at(0), ipoint, direction_vector);
        possible_points.at(1) = linesIntersection(primitive_vertices.at(2), primitive_vertices.at(2) - primitive_vertices.at(1), ipoint, direction_vector);
        possible_points.at(2) = linesIntersection(primitive_vertices.at(0), primitive_vertices.at(0) - primitive_vertices.at(2), ipoint, direction_vector);
        
        for(uint i=0;i<3;i++)
        {
            if(i == 0)
            {
                vi1 = primitive_vertices.at(1);
                vi0 = primitive_vertices.at(0);
            }else if(i == 1){
                vi1 = primitive_vertices.at(2);
                vi0 = primitive_vertices.at(1);
                
            }else if(i == 2){
                vi1 = primitive_vertices.at(0);
                vi0 = primitive_vertices.at(2);
            }
            
            bool test_x = ( (possible_points.at(i).x >= vi1.x && possible_points.at(i).x <= vi0.x) || (possible_points.at(i).x >= vi0.x && possible_points.at(i).x <= vi1.x) );
            bool test_y = ( (possible_points.at(i).y >= vi1.y && possible_points.at(i).y <= vi0.y) || (possible_points.at(i).y >= vi0.y && possible_points.at(i).y <= vi1.y) );
            bool test_z = ( (possible_points.at(i).z >= vi1.z && possible_points.at(i).z <= vi0.z) || (possible_points.at(i).z >= vi0.z && possible_points.at(i).z <= vi1.z) );
            
            if( test_x && test_y && test_z){
                slice_points.push_back(possible_points.at(i));
                slice_points_edge_ID.push_back(i);
            }
        }
        
    }
    

    // can be 0, 1, 2, 3, or 4 (0 and 2 are most common)
    uint initial_slice_points_size = slice_points.size();
    // std::cout << "initial_slice_points_size = " << initial_slice_points_size << std::endl;

    float absTol = pow(10, -6);
    float relTol = pow(10, -20);
    
    
    // the primitive did not intersect with the voxel face
    if(initial_slice_points_size == 0){
        resulting_UUIDs.push_back(UUID);
        return resulting_UUIDs;
        
    }else if(initial_slice_points_size == 1){
        // the primitive intersected with the face at a single point (a corner) - no slicing needed
        resulting_UUIDs.push_back(UUID);
        if(printmessages){
            std::cout << "the primitive intersected with the face at a single point (a corner) - no slicing needed" << std::endl;
        }
        return resulting_UUIDs;
    }else if(initial_slice_points_size == 2){
        
        // This is the usual case
        // just check to see if the two slice points are approximately at two vertices for edge cases here
        
        // the primitive intersected with the face along an edge - no need to slice
        if(slice_points_edge_ID.at(0) == slice_points_edge_ID.at(1)){
            resulting_UUIDs.push_back(UUID);
            if(printmessages){
                std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
            }
            return resulting_UUIDs;
        }
        
        if(primitive_vertices.size() == 4)
        {
            if((approxSame(slice_points.at(0), primitive_vertices.at(0), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(1), absTol))  ||
               (approxSame(slice_points.at(0), primitive_vertices.at(1), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(0), absTol)) ||
               (approxSame(slice_points.at(0), primitive_vertices.at(1), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(2), absTol)) ||
               (approxSame(slice_points.at(0), primitive_vertices.at(2), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(1), absTol)) ||
               (approxSame(slice_points.at(0), primitive_vertices.at(2), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(3), absTol)) ||
               (approxSame(slice_points.at(0), primitive_vertices.at(3), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(2), absTol)) ||
               (approxSame(slice_points.at(0), primitive_vertices.at(3), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(0), absTol)) ||
               (approxSame(slice_points.at(0), primitive_vertices.at(0), absTol) && approxSame(slice_points.at(1), primitive_vertices.at(3), absTol)) )
            {
                if(printmessages){
                    std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
                }
                resulting_UUIDs.push_back(UUID);
                return resulting_UUIDs;
            }
            
        }else if(primitive_vertices.size() == 3){
            
            if((approxSame(slice_points.at(0), primitive_vertices.at(0), absTol) || approxSame(slice_points.at(0), primitive_vertices.at(1), absTol) ||
               approxSame(slice_points.at(0), primitive_vertices.at(2), absTol)) &&
               (approxSame(slice_points.at(1), primitive_vertices.at(0), absTol) || approxSame(slice_points.at(1), primitive_vertices.at(1), absTol) ||
               approxSame(slice_points.at(1), primitive_vertices.at(2), absTol))){
                resulting_UUIDs.push_back(UUID);
                if(printmessages){
                    std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
                }
                return resulting_UUIDs;
            }
        }
        

        // now that edge cases are taken care of,
        // for each slice point, if it is approximately the same as a vertex, set it to that vertex
        for(uint j = 0;j<primitive_vertices.size();j++)
        {
            for(uint i=0;i<slice_points.size();i++)
            {
                //distance between slice point and primitive vertex
                float Dxyza = sqrtf(powf(primitive_vertices.at(j).x - slice_points.at(i).x, 2.0) + powf(primitive_vertices.at(j).y - slice_points.at(i).y, 2.0) + powf(primitive_vertices.at(j).z - slice_points.at(i).z, 2.0) ); 
                if(approxSame(Dxyza, float(0.0), absTol, relTol))
                {
                    slice_points.at(i) = primitive_vertices.at(j);
                }
            }  
        }
            
        
    }else if(initial_slice_points_size == 3){
        
        //if there are 3 slice points, this probably means that two of the points are very close to each other,
        // at or approximately at one of the primitive's vertices
        // in this case, if the primitive is a triangle, then it should be sliced into two triangles, not the usual three
        // in case the primitive is a patch, then it should be sliced into 3 triangles if this occurs at only one vertex
        
        vec3 non_vertex_slice_point;
        uint non_vertex_slice_edge_ID;
        vec3 vertex_slice_point;
        
        for(uint bb=0;bb < slice_points.size();bb++)
        {
            bool this_point_vert_test = false;
            for(uint cc=0;cc < primitive_vertices.size();cc++)
            {
                bool vert_test = approxSame(slice_points.at(bb), primitive_vertices.at(cc), absTol);
                 // std::cout << "-- test = " << vert_test <<" -- slice point " << bb << " = " << slice_points.at(bb) << ", primitive_vertex " << cc << " = " << primitive_vertices.at(cc) << std::endl;
                if(vert_test)
                {
                    this_point_vert_test = true;
                    vertex_slice_point = primitive_vertices.at(cc);
                    vertex_index = cc;
                }
            }
            
            if(this_point_vert_test == false)
            {
                non_vertex_slice_point = slice_points.at(bb); 
                non_vertex_slice_edge_ID = slice_points_edge_ID.at(bb);
            }
            
        }
        slice_points.resize(2);
        slice_points.at(0) = non_vertex_slice_point;
        slice_points_edge_ID.at(0) = non_vertex_slice_edge_ID;
        slice_points.at(1) = vertex_slice_point;
        
        // std::cout << "slice_points.at(0) = " << slice_points.at(0) << std::endl;
        // std::cout << "slice_points.at(1) = " << slice_points.at(1) << std::endl;
        // std::cout << "slice_points_edge_ID.at(0) = " << slice_points_edge_ID.at(0) << std::endl;
        // std::cout << "vertex_index = " << vertex_index << std::endl;
        
    }else if(initial_slice_points_size == 4){
        // if the voxel face splits a patch diagonally, then only 2 triangles should be produced instead of the usual four
        vec3 non_vertex_slice_point;
        uint non_vertex_slice_edge_ID;
        vec3 vertex_slice_point;
        for(uint bb=0;bb < slice_points.size();bb++)
        {
            bool this_point_vert_test = false;
            for(uint cc=0;cc < primitive_vertices.size();cc++)
            {
                bool vert_test = approxSame(slice_points.at(bb), primitive_vertices.at(cc), absTol);
                // std::cout << "-- test = " << vert_test <<" -- slice point " << bb << " = " << slice_points.at(bb) << ", primitive_vertex " << cc << " = " << primitive_vertices.at(cc) << std::endl;
                if(vert_test)
                {
                    this_point_vert_test = true;
                    vertex_index = cc;
                }
            }
        }
        slice_points.resize(2);
    }else{
        std::cerr << "ERROR (slicePrimitive): more than 5 slice points: something is very wrong..." << std::endl;
        throw(1);
    }
    
    //determine which side of the plane vertex 0 is on and use that to determine the sign of the buffer to add to the face coordinate
    // the buffer ensures that the vertices will be categorized into grid cells correctly
    // note that some of these checks are based on the assumption of a axis aligned grid - would need to be re-worked if implementing rotated grid
    
    helios::vec3 face_coordinate = make_vec3(fabs(face_normal.x)*voxel_face_vertices.at(0).x,fabs(face_normal.y)*voxel_face_vertices.at(0).y, fabs(face_normal.z)*voxel_face_vertices.at(0).z) ;
    // float buffer_value = powf(float(10), float(-6));
    float buffer_value = powf(float(10), float(-5));
    helios::vec3 buffer = make_vec3(0,0,0);
    if(fabs(face_normal.x) > 0.5)
    {
        if(primitive_vertices.at(0).x < face_coordinate.x)
        {
            buffer = make_vec3(float(-1)*buffer_value, 0, 0);
        }else if(primitive_vertices.at(0).x > face_coordinate.x){
            buffer = make_vec3(buffer_value, 0, 0);
        }else{
            if(printmessages){
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }
        
    }else if(fabs(face_normal.y) > 0.5){
        if(primitive_vertices.at(0).y < face_coordinate.y )
        {
            buffer = make_vec3( 0,float(-1)*buffer_value, 0);
        }else if(primitive_vertices.at(0).y > face_coordinate.y){
            buffer = make_vec3( 0,buffer_value, 0);
        }else{
            if(printmessages){
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }
        
    }else if(fabs(face_normal.z) > 0.5){
        if(primitive_vertices.at(0).z < face_coordinate.z )
        {
            buffer = make_vec3( 0, 0, float(-1)*buffer_value);
        }else if(primitive_vertices.at(0).z > face_coordinate.z){
            buffer = make_vec3(0, 0, buffer_value);
        } else{
            if(printmessages){
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }
    }
    
    // UUIDs for triangles to be created below
    uint t0;
    uint t1;
    uint t2;
    uint t3;
    
    //if a resulting triangle area is below this value, delete it
    float minArea = pow(10, -13);
    
    // use this diagnostic code to locate where a particular triangle is being created
    // (uncomment the print out far below)
    uint diag_1 = 0;
    
    /////////////////////////////////////////////////////////////////////////////////////////////
    // if the primitive isn't texture masked
    if(primitiveHasTexture == false){
        
        if(primitive_vertices.size() == 3)
        {
            // split into three triangles (usual case)
            if(initial_slice_points_size == 2)
            {
                if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1))
                {
                    diag_1 = 1;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 2;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 3;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 4;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 5;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);//
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);//
                    t2 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 6;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);//
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);//
                    t2 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                }
                if(context->getPrimitiveArea(t2) < minArea)
                {
                    context->deletePrimitive(t2);
                }else{
                    resulting_UUIDs.push_back(t2);
                }
                
                   
            }else if(initial_slice_points_size == 3){
                // split into two triangles instead of three since a vertex falls on the slicing face
                
                if(slice_points_edge_ID.at(0) == 0)
                {
                    diag_1 = 7;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 1){
                    diag_1 = 8;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 2){
                    diag_1 = 9;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                } 
            }
            
        }else if( primitive_vertices.size() == 4){
            
            // split into four triangles (usual case) 
            if(initial_slice_points_size == 2)
            {
                //cases where intersection points are on opposite sides
                if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2))
                {
                    diag_1 = 10;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    
                    
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 11;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 3)){
                    diag_1 = 12;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    
                }else if((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 13;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    
                    //cases where intersection points are on adjacent sides
                }else if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 3)){
                    diag_1 = 14;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 15;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer,  primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    
                }else if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 16;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 17;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 18;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 19;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 20;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 3)){
                    diag_1 = 21;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t3 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_color);
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                }
                if(context->getPrimitiveArea(t2) < minArea)
                {
                    context->deletePrimitive(t2);
                }else{
                    resulting_UUIDs.push_back(t2);
                }
                if(context->getPrimitiveArea(t3) < minArea)
                {
                    context->deletePrimitive(t3);
                }else{
                    resulting_UUIDs.push_back(t3);
                }
                
            }else if(initial_slice_points_size == 3){
                // split into three triangles instead of four since one vertex falls on the slicing face
                
                if(slice_points_edge_ID.at(0) == 0 && vertex_index == 2)
                {
                    diag_1 = 22;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer,  primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 0 && vertex_index == 3){
                    diag_1 = 23;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer,  primitive_vertices.at(0) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) - buffer,  primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 1 && vertex_index == 3){
                    diag_1 = 24;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer,  primitive_vertices.at(1) + buffer,  primitive_color);
                    t1 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 1 && vertex_index == 0){
                    diag_1 = 25;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) - buffer,  primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 2 && vertex_index == 1){
                    diag_1 = 26;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer,  primitive_vertices.at(2) - buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 2 && vertex_index == 0){
                    diag_1 = 27;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) - buffer,  primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 3 && vertex_index == 2){
                    diag_1 = 28;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(0) - buffer,   slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                }else if(slice_points_edge_ID.at(0) == 3 && vertex_index == 1){
                    diag_1 = 29;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, primitive_color);
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                    t2 = context->addTriangle(slice_points.at(1) - buffer,  primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, primitive_color);
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                } 
                if(context->getPrimitiveArea(t2) < minArea)
                {
                    context->deletePrimitive(t2);
                }else{
                    resulting_UUIDs.push_back(t2);
                } 
                
            }else if(initial_slice_points_size == 4){
                // split into two triangles instead of four since both vertices fall on the slicing face
                if(vertex_index == 0 || vertex_index == 2)
                {
                    diag_1 = 30;
                    t0 = context->addTriangle(primitive_vertices.at(0) - buffer, primitive_vertices.at(1) - buffer,  primitive_vertices.at(2) - buffer, primitive_color);
                    t1 = context->addTriangle(primitive_vertices.at(0) + buffer,  primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, primitive_color);
                  
                }else if(vertex_index == 1 || vertex_index == 3){  
                    diag_1 = 31;
                    t0 = context->addTriangle(primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer,  primitive_vertices.at(3) - buffer, primitive_color);
                    t1 = context->addTriangle(primitive_vertices.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_color);
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                } 
                
            }
        }
        
    }else if(primitiveHasTexture){
        
        //get uv coordinates of the vertices
        std::vector<helios::vec2> v_uv = context->getPrimitiveTextureUV(UUID);
        
        //get uv coordinates of the intersection points
        std::vector<helios::vec2> ip_uv;
        ip_uv.resize(2);
        
        if(primitive_vertices.size() == 3){
            
            // split into three triangles (usual case) 
            if(initial_slice_points_size == 2)
            {
                for(uint i=0;i<slice_points.size();i++)
                {
                    //vectors to hold point coordinates and uv coordinates for the points on the current point's edge
                    helios::vec3 point_0;
                    helios::vec3 point_1;
                    helios::vec2 point_0uv;
                    helios::vec2 point_1uv;
                    helios::vec2 point_uv;
                    
                    if(slice_points_edge_ID.at(i) == 0)
                    {
                        point_0 = primitive_vertices.at(0);
                        point_1 = primitive_vertices.at(1);
                        point_0uv = v_uv.at(0);
                        point_1uv = v_uv.at(1);
                    }else if(slice_points_edge_ID.at(i) == 1){
                        point_0 = primitive_vertices.at(1);
                        point_1 = primitive_vertices.at(2);
                        point_0uv = v_uv.at(1);
                        point_1uv = v_uv.at(2);
                    }else if(slice_points_edge_ID.at(i) == 2){
                        point_0 = primitive_vertices.at(2);
                        point_1 = primitive_vertices.at(0);
                        point_0uv = v_uv.at(2);
                        point_1uv = v_uv.at(0);
                    }
                    
                    ip_uv.at(i) =  interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(i));

                    if(ip_uv.at(0).x < 0 || ip_uv.at(0).x > 1 || ip_uv.at(0).y < 0 || ip_uv.at(0).y > 1)
                    {
                        std::cerr << "ERROR in slicePrimitive: texture uv for UUID " << UUID << " < 0 or > 1" << std::endl;
                        throw(1);
                    }
                }
                
                if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1))
                {
                    diag_1 = 101;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0), v_uv.at(1), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                    t2 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(0));
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 102;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(2), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1) ,v_uv.at(2), v_uv.at(0));
                }else if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 103;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(2), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 104;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(0), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                    t2 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 105;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(2), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 106;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(0), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                }
                if(context->getPrimitiveArea(t2) < minArea)
                {
                    context->deletePrimitive(t2);
                }else{
                    resulting_UUIDs.push_back(t2);
                }
                
                // split into two triangles instead of three since a vertex falls on the slicing face
                // the non-vertex slice point is slice_points.at(0) and the vertex slice point is slice_points.at(1)
            }else if(initial_slice_points_size == 3){
                
                // std::cout << "initial_slice_points_size = " << initial_slice_points_size << std::endl;
                
                //vectors to hold point coordinates and uv coordinates for the points on the current point's edge for interpolation
                helios::vec3 point_0;
                helios::vec3 point_1;
                helios::vec2 point_0uv;
                helios::vec2 point_1uv;
                helios::vec2 point_uv;
                
                if(slice_points_edge_ID.at(0) == 0)
                {
                    point_0 = primitive_vertices.at(0);
                    point_1 = primitive_vertices.at(1);
                    point_0uv = v_uv.at(0);
                    point_1uv = v_uv.at(1);
                    ip_uv.at(1) = v_uv.at(2); // this sets the uv coordinate for the vertex slice point
                    
                }else if(slice_points_edge_ID.at(0) == 1){
                    point_0 = primitive_vertices.at(1);
                    point_1 = primitive_vertices.at(2);
                    point_0uv = v_uv.at(1);
                    point_1uv = v_uv.at(2);
                    ip_uv.at(1) = v_uv.at(0); // this sets the uv coordinate for the vertex slice point
                }else if(slice_points_edge_ID.at(0) == 2){
                    point_0 = primitive_vertices.at(2);
                    point_1 = primitive_vertices.at(0);
                    point_0uv = v_uv.at(2);
                    point_1uv = v_uv.at(0);
                    ip_uv.at(1) = v_uv.at(1); // this sets the uv coordinate for the vertex slice point
                }
                
                // UV for non-vertex slice point
                ip_uv.at(0) =  interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(0));
                
                if(ip_uv.at(0).x < 0 || ip_uv.at(0).x > 1 || ip_uv.at(0).y < 0 || ip_uv.at(0).y > 1)
                {
                    std::cerr << "ERROR in slicePrimitive: texture uv for UUID " << UUID << " < 0 or > 1" << std::endl;
                    throw(1);
                }
                
                if(slice_points_edge_ID.at(0) == 0)
                {
                    diag_1 = 107;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer,  tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer,  tex, ip_uv.at(0) ,v_uv.at(1), ip_uv.at(1));
                }else if(slice_points_edge_ID.at(0) == 1){
                    diag_1 = 108;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(2) + buffer, slice_points.at(1) + buffer,  tex, ip_uv.at(0), v_uv.at(2), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer,  tex, ip_uv.at(0), ip_uv.at(1) , v_uv.at(1));
                }else if(slice_points_edge_ID.at(0) == 2){
                    diag_1 = 109;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer,  tex, ip_uv.at(0), ip_uv.at(1) , v_uv.at(2));
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                }
            }
            
            
        }else if(primitive_vertices.size() == 4){
            
            // it seems patches that are not explicitly set up with texture UV coordinates just don't have them
            // so set the default here
            if(v_uv.size() == 0)
            {
                std::vector<helios::vec2> uv{ make_vec2(0,0), make_vec2(1,0), make_vec2(1,1), make_vec2(0,1) };
                v_uv = uv;
            }
            
            // split into four triangles (usual case) 
            if(initial_slice_points_size == 2)
            {
                // for each intersection point, choose the patch vertices on the corresponding edge
                for(uint i=0;i<2;i++)
                {
                    helios::vec3 point_0;
                    helios::vec3 point_1;
                    helios::vec2 point_uv;
                    helios::vec2 point_0uv;
                    helios::vec2 point_1uv;
                    
                    if(slice_points_edge_ID.at(i) == 0)
                    {
                        point_0 = primitive_vertices.at(0);
                        point_1 = primitive_vertices.at(1);
                        point_0uv = v_uv.at(0);
                        point_1uv = v_uv.at(1);
                    }else if(slice_points_edge_ID.at(i) == 1){
                        point_0 = primitive_vertices.at(1);
                        point_1 = primitive_vertices.at(2);
                        point_0uv = v_uv.at(1);
                        point_1uv = v_uv.at(2);
                    }else if(slice_points_edge_ID.at(i) == 2){
                        point_0 = primitive_vertices.at(2);
                        point_1 = primitive_vertices.at(3);
                        point_0uv = v_uv.at(2);
                        point_1uv = v_uv.at(3);
                    }else if(slice_points_edge_ID.at(i) == 3){
                        point_0 = primitive_vertices.at(3);
                        point_1 = primitive_vertices.at(0);
                        point_0uv = v_uv.at(3);
                        point_1uv = v_uv.at(0);
                    }
                    
                    ip_uv.at(i) =  interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(i));
                    
                    if(ip_uv.at(i).x < 0 || ip_uv.at(i).x > 1 || ip_uv.at(i).y < 0 || ip_uv.at(i).y > 1)
                    {
                        std::cerr << "ERROR in slicePrimitive: texture uv for UUID " << UUID << " < 0 or > 1" << std::endl;
                        throw(1);
                    }
                }
                
                //cases where intersection points are on opposite sides
                if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 2))
                {
                    diag_1 = 110;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer,  primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(2), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(3) ,v_uv.at(0));
                    t3 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 111;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(0), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3) ,v_uv.at(0));
                    t3 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 3)){
                    diag_1 = 112;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(3), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
                    t3 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2) ,v_uv.at(3));
                }else if((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 113;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(1), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
                    t3 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(1), v_uv.at(2) ,v_uv.at(3));
                    //cases where intersection points are on adjacent sides
                }else if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 3)){
                    diag_1 = 114;
                    t0 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(2), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
                    t3 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(1), v_uv.at(2) ,v_uv.at(3));
                }else if((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 115;
                    t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(0), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                    t2 = context->addTriangle(slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
                    t3 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2) ,v_uv.at(3));
                }else if((slice_points_edge_ID.at(0) == 0 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 116;
                    t0 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(1), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, tex, ip_uv.at(1), v_uv.at(2) ,v_uv.at(3));
                    t3 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0) ,v_uv.at(3), v_uv.at(0));
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 0)){
                    diag_1 = 117;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(3), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, tex, ip_uv.at(0), v_uv.at(2) ,v_uv.at(3));
                    t3 = context->addTriangle(slice_points.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1) ,v_uv.at(3), v_uv.at(0));
                }else if((slice_points_edge_ID.at(0) == 1 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 118;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(2), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(03) ,v_uv.at(0));
                    t3 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 1)){
                    diag_1 = 119;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(0), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3) ,v_uv.at(0));
                    t3 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
                }else if((slice_points_edge_ID.at(0) == 3 && slice_points_edge_ID.at(1) == 2)){
                    diag_1 = 120;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, slice_points.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                    t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(1) + buffer, slice_points.at(1) + buffer, tex, ip_uv.at(0) ,v_uv.at(1), ip_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
                    t3 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
                }else if((slice_points_edge_ID.at(0) == 2 && slice_points_edge_ID.at(1) == 3)){
                    diag_1 = 121;
                    t0 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(3) - buffer, slice_points.at(1) - buffer, tex, ip_uv.at(0) ,v_uv.at(3), ip_uv.at(1));
                    t1 = context->addTriangle(slice_points.at(0) + buffer, slice_points.at(1) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                    t2 = context->addTriangle(slice_points.at(1) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
                    t3 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(1) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                }
                if(context->getPrimitiveArea(t2) < minArea)
                {
                    context->deletePrimitive(t2);
                }else{
                    resulting_UUIDs.push_back(t2);
                }
                if(context->getPrimitiveArea(t3) < minArea)
                {
                    context->deletePrimitive(t3);
                }else{
                    resulting_UUIDs.push_back(t3);
                }
                
                
            }else if(initial_slice_points_size == 3){
                
                // for the first intersection point (index 0), choose the endpoints of the edge to interpolate UV between
                // for this case where the other intersection point is at a primitive vertex, that vertex UV will be used
                for(uint i=0;i<1;i++)
                {
                    helios::vec3 point_0;
                    helios::vec3 point_1;
                    helios::vec2 point_uv;
                    helios::vec2 point_0uv;
                    helios::vec2 point_1uv;
                    
                    if(slice_points_edge_ID.at(i) == 0)
                    {
                        point_0 = primitive_vertices.at(0);
                        point_1 = primitive_vertices.at(1);
                        point_0uv = v_uv.at(0);
                        point_1uv = v_uv.at(1);
                    }else if(slice_points_edge_ID.at(i) == 1){
                        point_0 = primitive_vertices.at(1);
                        point_1 = primitive_vertices.at(2);
                        point_0uv = v_uv.at(1);
                        point_1uv = v_uv.at(2);
                    }else if(slice_points_edge_ID.at(i) == 2){
                        point_0 = primitive_vertices.at(2);
                        point_1 = primitive_vertices.at(3);
                        point_0uv = v_uv.at(2);
                        point_1uv = v_uv.at(3);
                    }else if(slice_points_edge_ID.at(i) == 3){
                        point_0 = primitive_vertices.at(3);
                        point_1 = primitive_vertices.at(0);
                        point_0uv = v_uv.at(3);
                        point_1uv = v_uv.at(0);
                    }
                    
                    // std::cout << "point_0 = " << point_0 << std::endl;
                    // std::cout << "point_0uv = " << point_0uv << std::endl;
                    // std::cout << "point_1 = " << point_1 << std::endl;
                    // std::cout << "point_1uv = " << point_1uv << std::endl;
                    // std::cout << "i = " <<  i << std::endl;
                    // std::cout << "slice_points.at(i) = " << slice_points.at(i) << std::endl;
                    // std::cout << "slice_points.size() = " << slice_points.size() << std::endl;
                    // std::cout << "slice_points_edge_ID.at(i) = " << slice_points_edge_ID.at(i) << std::endl;
                    
                    ip_uv.at(i) =  interpolate_texture_UV_to_slice_point(point_0, point_0uv, point_1, point_1uv, slice_points.at(i));
                    
                    if(ip_uv.at(i).x < 0 || ip_uv.at(i).x > 1 || ip_uv.at(i).y < 0 || ip_uv.at(i).y > 1)
                    {
                        std::cerr << "ERROR in slicePrimitive: texture uv for UUID " << UUID << " < 0 or > 1" << std::endl;
                        throw(1);
                    }
                
        
                    if(slice_points_edge_ID.at(0) == 0 && vertex_index == 2)
                    {
                        diag_1 = 122;
                        t0 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(2) + buffer,  primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(0));
                        t1 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                        t2 = context->addTriangle(primitive_vertices.at(2) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, v_uv.at(2), v_uv.at(3), v_uv.at(0));
                    }else if(slice_points_edge_ID.at(0) == 0 && vertex_index == 3){
                        diag_1 = 123;
                        t0 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer,  primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                        t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0) , v_uv.at(1), v_uv.at(3));
                        t2 = context->addTriangle(primitive_vertices.at(3) - buffer,  primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, v_uv.at(3), v_uv.at(1) , v_uv.at(2));
                    }else if(slice_points_edge_ID.at(0) == 1 && vertex_index == 3){
                        diag_1 = 124;
                        t0 = context->addTriangle(slice_points.at(0) + buffer, primitive_vertices.at(3) + buffer,  primitive_vertices.at(1) + buffer,  tex, ip_uv.at(0), v_uv.at(3) , v_uv.at(1));
                        t1 = context->addTriangle(primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, v_uv.at(3), v_uv.at(0), v_uv.at(1));
                        t2 = context->addTriangle(slice_points.at(0) - buffer,  primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                    }else if(slice_points_edge_ID.at(0) == 1 && vertex_index == 0){
                        diag_1 = 125;
                        t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1));
                        t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(2) - buffer, primitive_vertices.at(0) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(0));
                        t2 = context->addTriangle(primitive_vertices.at(0) - buffer,  primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, v_uv.at(0), v_uv.at(2), v_uv.at(3));
                    }else if(slice_points_edge_ID.at(0) == 2 && vertex_index == 1){
                        diag_1 = 126;
                        t0 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer,  primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1), v_uv.at(2));
                        t1 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(1));
                        t2 = context->addTriangle(primitive_vertices.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, v_uv.at(1), v_uv.at(3) , v_uv.at(0));
                    }else if(slice_points_edge_ID.at(0) == 2 && vertex_index == 0){
                        diag_1 = 127;
                        t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(3), v_uv.at(0));
                        t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(0) - buffer, primitive_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(2));
                        t2 = context->addTriangle(primitive_vertices.at(0) - buffer,  primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer, tex, v_uv.at(0), v_uv.at(1), v_uv.at(2));
                    }else if(slice_points_edge_ID.at(0) == 3 && vertex_index == 2){
                        diag_1 = 128;
                        t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, primitive_vertices.at(2) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(2));
                        t1 = context->addTriangle(primitive_vertices.at(2) + buffer, primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, v_uv.at(2), v_uv.at(0), v_uv.at(1));
                        t2 = context->addTriangle(slice_points.at(0) - buffer,   primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2), v_uv.at(3));
                    }else if(slice_points_edge_ID.at(0) == 3 && vertex_index == 1){
                        diag_1 = 129;
                        t0 = context->addTriangle(slice_points.at(0) + buffer,  primitive_vertices.at(0) + buffer, primitive_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0), v_uv.at(1) );
                        t1 = context->addTriangle(slice_points.at(0) - buffer, primitive_vertices.at(1) - buffer, primitive_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(1)  ,v_uv.at(3));
                        t2 = context->addTriangle(primitive_vertices.at(1) - buffer,  primitive_vertices.at(2) - buffer, primitive_vertices.at(3) - buffer, tex, v_uv.at(1) , v_uv.at(2), v_uv.at(3));
                    }
                    
                    //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                    if(context->getPrimitiveArea(t0) < minArea)
                    {
                        context->deletePrimitive(t0);
                    }else{
                        resulting_UUIDs.push_back(t0);
                    }
                    if(context->getPrimitiveArea(t1) < minArea)
                    {
                        context->deletePrimitive(t1);
                    }else{
                        resulting_UUIDs.push_back(t1);
                    } 
                    if(context->getPrimitiveArea(t2) < minArea)
                    {
                        context->deletePrimitive(t2);
                    }else{
                        resulting_UUIDs.push_back(t2);
                    }
                }
                
            }else if(initial_slice_points_size == 4){
                
                if(vertex_index == 0 || vertex_index == 2)
                {
                    diag_1 = 130;
                    t0 = context->addTriangle(primitive_vertices.at(0) - buffer, primitive_vertices.at(1) - buffer,  primitive_vertices.at(2) - buffer, tex, v_uv.at(0), v_uv.at(1), v_uv.at(2));
                    t1 = context->addTriangle(primitive_vertices.at(0) + buffer,  primitive_vertices.at(2) + buffer, primitive_vertices.at(3) + buffer, tex, v_uv.at(0), v_uv.at(2), v_uv.at(3));
                    
                }else if(vertex_index == 1 || vertex_index == 3){  
                    diag_1 = 131;
                    t0 = context->addTriangle(primitive_vertices.at(1) - buffer, primitive_vertices.at(2) - buffer,  primitive_vertices.at(3) - buffer, tex, v_uv.at(1), v_uv.at(2), v_uv.at(3));
                    t1 = context->addTriangle(primitive_vertices.at(1) + buffer,  primitive_vertices.at(3) + buffer, primitive_vertices.at(0) + buffer, tex, v_uv.at(1), v_uv.at(3), v_uv.at(0));
                }
                
                //delete triangles with area of zero, otherwise add to resulting_UUIDs vector
                if(context->getPrimitiveArea(t0) < minArea)
                {
                    context->deletePrimitive(t0);
                }else{
                    resulting_UUIDs.push_back(t0);
                }
                if(context->getPrimitiveArea(t1) < minArea)
                {
                    context->deletePrimitive(t1);
                }else{
                    resulting_UUIDs.push_back(t1);
                } 
            }
        }
    }
    
    // print this out to find where a certain triangle is created
    // std::cout << "diag_1 = " << diag_1 << std::endl;
    
    // copy over primitive data to the new triangles
    for(uint i=0;i<resulting_UUIDs.size();i++){
        context->copyPrimitiveData(UUID, resulting_UUIDs.at(i));
        uint parentID = context->getPrimitiveParentObjectID(UUID);
        if( parentID>0 && context->getObjectType(parentID)==helios::OBJECT_TYPE_TILE ){
            context->setPrimitiveParentObjectID(resulting_UUIDs.at(i),0);
        }else {
            context->setPrimitiveParentObjectID(resulting_UUIDs.at(i), parentID);
        }
        if( context->isPrimitiveTextureColorOverridden(UUID) ){
            context->overridePrimitiveTextureColor(resulting_UUIDs.at(i));
        }
    }
    
    // compare original and resulting primitive areas to make sure they approximately match
    float resulting_area = context->sumPrimitiveSurfaceArea(resulting_UUIDs);
    float pdiff_area = (resulting_area - original_area)/original_area*100.0;
    float pdiff_area_abs = fabs(pdiff_area);
    if(pdiff_area_abs > 1)
    {
        std::cout << "WARNING: sum of slice areas does not equal area of original primitive (UUID = " << UUID << ")" << std::endl;
        std::cout << "original area = " << original_area << std::endl;
        std::cout << "resulting_area = " << resulting_area << std::endl;
        std::cout << "pdiff_area = " << pdiff_area << std::endl;
        std::cout << "resulting_UUIDs.size() = " << resulting_UUIDs.size() << std::endl; 
    }
    
    // compare original and resulting primitive normals to make sure they match
    absTol = 0.5;
    relTol = 0.4;
    for(uint aa=0;aa<resulting_UUIDs.size();aa++)
    {
        helios::vec3 this_normal = context->getPrimitiveNormal(resulting_UUIDs.at(aa));
        this_normal.normalize(); 
        if(!approxSame(primitive_normal.x, this_normal.x, absTol, relTol) || !approxSame(primitive_normal.y, this_normal.y, absTol, relTol)  || !approxSame(primitive_normal.z, this_normal.z, absTol, relTol) )
        {
            std::cout << "WARNING: UUID " << resulting_UUIDs.at(aa) << " normal " << this_normal << " does not match original normal " << primitive_normal << std::endl;
        }
    }
    
    // delete the original primitive
    context->deletePrimitive(UUID);
    
    return resulting_UUIDs;
}

std::vector<uint> VoxelIntersection::slicePrimitivesUsingGrid(std::vector<uint> UUIDs, helios::vec3 grid_center, helios::vec3 grid_size, helios::int3 grid_divisions)
{
    
    //set up the grid
    std::vector<std::vector<helios::vec3>> grid_face_vertices;
    helios::vec3 grid_min = make_vec3(grid_center.x - grid_size.x*0.5, grid_center.y - grid_size.y*0.5, grid_center.z - grid_size.z*0.5);
    helios::vec3 grid_max = make_vec3(grid_center.x + grid_size.x*0.5, grid_center.y + grid_size.y*0.5, grid_center.z + grid_size.z*0.5);
    helios::vec3 grid_spacing = make_vec3(grid_size.x/grid_divisions.x, grid_size.y/grid_divisions.y, grid_size.z/grid_divisions.z);
    
    //faces in the y-z plane (change x)
    for(uint k=0;k< (grid_divisions.x + 1); k++)
    {
        std::vector<helios::vec3> this_face_vertices;
        this_face_vertices.push_back(make_vec3(grid_min.x + k*grid_spacing.x, grid_min.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_min.x + k*grid_spacing.x, grid_min.y, grid_max.z));
        this_face_vertices.push_back(make_vec3(grid_min.x + k*grid_spacing.x, grid_max.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_min.x + k*grid_spacing.x, grid_max.y, grid_max.z));
        grid_face_vertices.push_back(this_face_vertices);
    }
    
    //faces in the x-z plane (change y)
    for(uint k=0;k< (grid_divisions.y + 1); k++)
    {
        std::vector<helios::vec3> this_face_vertices;
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_min.y + k*grid_spacing.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_min.y + k*grid_spacing.y, grid_max.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_min.y + k*grid_spacing.y, grid_min.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_min.y + k*grid_spacing.y, grid_max.z));
        grid_face_vertices.push_back(this_face_vertices);
    }
    
    //faces in the x-z plane (change y)
    for(uint k=0;k< (grid_divisions.z + 1); k++)
    {
        std::vector<helios::vec3> this_face_vertices;
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_min.y, grid_min.z + k*grid_spacing.z));
        this_face_vertices.push_back(make_vec3(grid_min.x, grid_max.y, grid_min.z + k*grid_spacing.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_min.y, grid_min.z + k*grid_spacing.z));
        this_face_vertices.push_back(make_vec3(grid_max.x, grid_max.y, grid_min.z + k*grid_spacing.z));
        grid_face_vertices.push_back(this_face_vertices);
    }
    
    if( printmessages ){
        std::cout << UUIDs.size() << " input primitives" << std::endl;
        std::cout << grid_face_vertices.size() << " grid faces used for slicing" << std::endl;
        std::cout << grid_divisions.x*grid_divisions.y*grid_divisions.z << " total grid cells" << std::endl;
    }
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // do an initial classification of primitives into grid cells based on if all their vertices fall into a given voxel
    
    cell_primitives.resize(grid_divisions.x*grid_divisions.y*grid_divisions.z + 1);
    
    //initially set all UUIDs as outside any voxel
    context->setPrimitiveData(UUIDs, "cell_ID", int(-1));
    
    //vectors for UUIDs that do and do not need to be sliced
    std::vector<uint> UUIDs_to_slice;
    std::vector<uint> UUIDs_no_slice;
    
    auto start = std::chrono::high_resolution_clock::now();
    for(uint p=0;p<UUIDs.size();p++)
    {
        int cell_ID = 0;
        bool flag = false;
        for(uint k=0;k< (grid_divisions.z); k++)
        {
            for(uint j=0;j< (grid_divisions.y); j++)
            {
                for(uint i=0;i< (grid_divisions.x); i++)
                {
                    helios::vec3 cell_min = make_vec3(grid_min.x + float(i)*grid_spacing.x, grid_min.y + float(j)*grid_spacing.y, grid_min.z + float(k)*grid_spacing.z);
                    helios::vec3 cell_max = make_vec3(grid_min.x + float(i)*grid_spacing.x + grid_spacing.x, grid_min.y + float(j)*grid_spacing.y + grid_spacing.y, grid_min.z + float(k)*grid_spacing.z + grid_spacing.z);
                    std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUIDs.at(p));
                    
                    uint v_in = 0;
                    for(uint v=0;v<verts.size();v++)
                    {
                        
                        bool test2_x = (verts.at(v).x >= cell_min.x) && (verts.at(v).x <= cell_max.x );
                        bool test2_y = (verts.at(v).y >= cell_min.y ) && (verts.at(v).y <= cell_max.y );
                        bool test2_z =  (verts.at(v).z >= cell_min.z ) && (verts.at(v).z <= cell_max.z ) ;
                        
                        if( test2_x && test2_y && test2_z)
                        {
                            v_in ++;
                        }
                    }
                    
                    if(v_in == verts.size())
                    {
                        //the UUID doesn't need to be sliced since its vertices all are within a cell
                        context->setPrimitiveData(UUIDs.at(p), "cell_ID", cell_ID);
                        cell_primitives.at(cell_ID).push_back(UUIDs.at(p));
                        UUIDs_no_slice.push_back(UUIDs.at(p));
                        cell_ID ++;
                        flag = true;
                        break;
                    }else if(v_in != 0){
                        //some verticies in and some out: UUID needs to be sliced
                        UUIDs_to_slice.push_back(UUIDs.at(p));
                        flag = true;
                        break;
                    }
                    cell_ID ++;
                }
                if(flag == true){break;}
            }
            if(flag == true){break;}
        }
        
        // if all vertices fell outside of all grid cells, add it to be sliced just in case (corner cases)
        if(flag == false){
            UUIDs_to_slice.push_back(UUIDs.at(p));
        }
        
    }
    
    if( printmessages ){
        
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
        std::cout << duration.count() << " seconds to do initial grid cell classification" << std::endl;
        std::cout << UUIDs_no_slice.size() << " input primitives (" << float(UUIDs_no_slice.size())/float(UUIDs.size())*100 << "%) not sliced" << std::endl;
        std::cout << UUIDs_to_slice.size() << " input primitives (" << float(UUIDs_to_slice.size())/float(UUIDs.size())*100 << "%) being sliced" << std::endl;
    }
    
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // do the slicing
    
    std::vector<uint> primitives_to_remove;
    std::vector<uint> primitives_to_add;
    uint s1;
    
    auto start2 = std::chrono::high_resolution_clock::now();
    //loop through each voxel face
    for(uint i=0;i<grid_face_vertices.size();i++)
    {
        for(uint j=0;j<UUIDs_to_slice.size();j++)
        {
            //slice
            std::vector<uint> resulting_UUIDs;
            resulting_UUIDs = slicePrimitive(UUIDs_to_slice.at(j), grid_face_vertices.at(i));
            
            //update the UUIDs_to_slice vector so it doesn't include deleted primitives (the originals that were split)
            bool exists = context->doesPrimitiveExist(UUIDs_to_slice.at(j));
            if(!exists)
            {
                primitives_to_remove.push_back(j);
                primitives_to_add.insert(primitives_to_add.end(), resulting_UUIDs.begin(), resulting_UUIDs.end());
            }
        }
        
        for(int k=primitives_to_remove.size()-1; k>=0; k--)
        {
            UUIDs_to_slice.erase(UUIDs_to_slice.begin() + primitives_to_remove.at(k));
        }
        primitives_to_remove.clear();
        
        UUIDs_to_slice.insert( UUIDs_to_slice.end(), primitives_to_add.begin(), primitives_to_add.end() );
        primitives_to_add.clear();
    }
    
    if( printmessages ){
        
        auto stop2 = std::chrono::high_resolution_clock::now();
        auto duration2 = std::chrono::duration_cast<std::chrono::seconds>(stop2 - start2);
        std::cout << duration2.count() << " seconds to do slicing" << std::endl;
        
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    // now classify the sliced primitives into grid cells
    // save the cell_ID as primitive data for the triangle
    // save the primitive UUID to a vector of UUIDs that are in a given cell and save that
    
    auto start3 = std::chrono::high_resolution_clock::now();
    
    for(uint p=0;p<UUIDs_to_slice.size();p++)
    {
        // std::cout << "UUIDs_to_slice.at(p) = " << UUIDs_to_slice.at(p) << std::endl;
        int cell_ID = 0;
        bool flag = false;
        
        for(uint k=0;k< (grid_divisions.z); k++)
        {
            for(uint j=0;j< (grid_divisions.y); j++)
            {
                for(uint i=0;i< (grid_divisions.x); i++)
                {
                    
                    helios::vec3 cell_min = make_vec3(grid_min.x + i*grid_spacing.x, grid_min.y + j*grid_spacing.y, grid_min.z + k*grid_spacing.z);
                    helios::vec3 cell_max = make_vec3(grid_min.x + i*grid_spacing.x + grid_spacing.x, grid_min.y + j*grid_spacing.y + grid_spacing.y, grid_min.z + k*grid_spacing.z + grid_spacing.z);
                    
                    std::vector<helios::vec3> verts = context->getPrimitiveVertices(UUIDs_to_slice.at(p));
                    uint v_in = 0;
                    for(uint v=0;v<verts.size();v++)
                    {
                        
                        float absTol = pow(10, -6);
                        float relTol = pow(10, -20);
                        bool test2_x = (verts.at(v).x > cell_min.x || approxSame(verts.at(v).x, cell_min.x, absTol, relTol)) && (verts.at(v).x < cell_max.x || approxSame(verts.at(v).x, cell_max.x, absTol, relTol));
                        bool test2_y = (verts.at(v).y > cell_min.y || approxSame(verts.at(v).y, cell_min.y, absTol, relTol)) && (verts.at(v).y < cell_max.y || approxSame(verts.at(v).y, cell_max.y, absTol, relTol));
                        bool test2_z =  (verts.at(v).z > cell_min.z || approxSame(verts.at(v).z, cell_min.z, absTol, relTol)) && (verts.at(v).z < cell_max.z || approxSame(verts.at(v).z, cell_max.z, absTol, relTol)) ;
                        
                        if( test2_x && test2_y && test2_z)
                        {
                            v_in ++;
                        }
                    }
                    
                    if(v_in == verts.size())
                    {
                        context->setPrimitiveData(UUIDs_to_slice.at(p), "cell_ID", cell_ID);
                        cell_primitives.at(cell_ID).push_back(UUIDs_to_slice.at(p));
                        cell_ID ++;
                        flag = true;
                        break;
                    }
                    cell_ID ++;
                }
                if(flag == true){break;}
            }
            if(flag == true){break;}
        }
        
        if(flag == false){
            cell_primitives.at(cell_primitives.size()-1).push_back(UUIDs_to_slice.at(p));
        }
        
    }
    
    if( printmessages ){
        auto stop3 = std::chrono::high_resolution_clock::now();
        auto duration3 = std::chrono::duration_cast<std::chrono::seconds>(stop3 - start3);
        std::cout << duration3.count() << " seconds to do second classification" << std::endl;
    }
    ///////////////////////////////////////////////////////////////////////////////////////////////////////
    //Join the unsliced and sliced primitive UUIDs back into a single vector
    std::vector<uint>  UUIDs_out = UUIDs_no_slice;
    UUIDs_out.insert(UUIDs_out.end(), UUIDs_to_slice.begin(), UUIDs_to_slice.end());
    
    if( printmessages ){
        std::cout << UUIDs_to_slice.size() << " primitives created from slicing" << std::endl;
        std::cout << UUIDs_out.size() << " total output primitives" << std::endl;
    }
    
    return UUIDs_out;
    
}


std::vector<std::vector<uint>> VoxelIntersection::getCellPrimitives(void){
         
  return cell_primitives;
  
}

bool VoxelIntersection::approxSame(float a, float b, float absTol, float relTol){
    
  return fabs(a-b) <= absTol || fabs(a-b) <= relTol*(std::max(fabs(a), fabs(b)));

}

bool VoxelIntersection::approxSame(helios::vec3 a, helios::vec3 b, float absTol)
{
    return fabs(a.x-b.x) <= absTol && fabs(a.y-b.y) <= absTol && fabs(a.z-b.z) <= absTol ;
}

helios::vec2 VoxelIntersection::interpolate_texture_UV_to_slice_point(helios::vec3 p1, helios::vec2 uv1, helios::vec3 p2, helios::vec2 uv2, helios::vec3 ps)
{
    // uv coordinate that will be output
    helios::vec2 uvs;
    
    float Dxyz = sqrtf(powf(p2.x - p1.x, 2.0) + powf(p2.y - p1.y, 2.0) + powf(p2.z - p1.z, 2.0) ); // distance between edge vertex xyz coordinates
    float Duv = sqrtf(powf(uv2.x - uv1.x, 2.0) + powf(uv2.y - uv1.y, 2.0) ); // distance between edge vertex uv coordinates
    float Dxyzs = sqrtf(powf(ps.x - p1.x, 2.0) + powf(ps.y - p1.y, 2.0)  + powf(ps.z - p1.z, 2.0) ); // distance between slice point and first vertex xyz coordinates

    // std::cout << "x1 = " << p1.x << " , y1 = " << p1.y << ", z1 = " << p1.z << std::endl;
    // std::cout << "x2 = " << p2.x << " , y2 = " << p2.y << ", z2 = " << p2.z << std::endl;
    // std::cout << "xs = " << ps.x << " , ys = " << ps.y << ", zs = " << ps.z << std::endl;
    // std::cout << "uv1 = " << uv1 << ", uv2 = " << uv2 << std::endl;
    // std::cout << "Dxyz = " << Dxyz << ", Duv = " << Duv << ", Dxyzs = " << Dxyzs << std::endl;
    // std::cout << "f = " << (Dxyzs/Dxyz) << std::endl;
    // std::cout << "f*Duv = " << Duv*(Dxyzs/Dxyz) << std::endl;

    float absTol = pow(10, -6);

    float F = (Dxyzs/Dxyz);
    if(F > 1.0)
    {
        std::cerr << "ERROR (interpolate_texture_UV_to_slice_point): slice point is not between the two end points" << std::endl;
        throw(1);
    }else if(approxSame(p1, ps, absTol)){
        // then the slice point is the same as the first vertex
        uvs = make_vec2(uv1.x, uv1.y);
        return uvs;
    }else if(approxSame(p2, ps, absTol)){
        // then the slice point is the same as the first vertex
        uvs = make_vec2(uv2.x, uv2.y);
        return uvs;
    }
    

    //if the u coordinates of the two vertices are the same
    if(uv2.x == uv1.x)
    {
        std::vector<float> vec_uv;
        vec_uv.push_back(uv1.y);
        vec_uv.push_back(uv2.y);
        uvs = make_vec2(uv1.x, min(vec_uv) + Duv*(Dxyzs/Dxyz)); 
        // std::cout << "uvs* = " << uvs << std::endl;
        
    }else{
        
        //equation for the line between uv coordinates of the two vertices
        float slope = (uv2.y - uv1.y)/(uv2.x - uv1.x);
        float offset = uv1.y - slope*uv1.x;
        
        // coefficients of the quadratic equation for the u coordinate of the slice point
        float a = powf(slope, 2.0) + 1.0;
        float b = -2.0*uv1.x + 2.0*slope*offset - 2.0*slope*uv1.y;
        float c = ( powf(uv1.x, 2.0) + powf(offset, 2.0) - 2.0*offset*uv1.y + powf(uv1.y, 2.0) ) - powf( (Dxyzs/Dxyz)*Duv, 2.0);
        
        // solve the quadratic
        float us_a = (-1.0*b + sqrtf(powf(b, 2.0) - 4.0*a*c))/(2.0*a);
        float us_b = (-1.0*b - sqrtf(powf(b, 2.0) - 4.0*a*c))/(2.0*a);
        // get the v coordinate   
        float vs_a = slope*us_a + offset;
        float vs_b = slope*us_b + offset;
        
        std::cout << "slope = " << slope << ", offset = " << offset << std::endl;
        std::cout << "a = " << a << ", b = " << b << ", c = " << c << std::endl;
        std::cout << "us_a = " << us_a << ", vs_a = " << vs_a << std::endl;
        std::cout << "us_b = " << us_b << ", vs_b = " << vs_b << std::endl;
        
        
        //determine which of the roots is the right one
        if( ((us_a >= uv1.x && us_a <= uv2.x) || (us_a <= uv1.x && us_a >= uv2.x)) && ((vs_a >= uv1.y && vs_a <= uv2.y) || (vs_a <= uv1.y && vs_a >= uv2.y)) )
        {
            uvs = make_vec2(us_a, vs_a); 
            // std::cout << "uvs = " << uvs << std::endl;
            
        }else if(((us_b >= uv1.x && us_b <= uv2.x) || (us_b <= uv1.x && us_b >= uv2.x)) && ((vs_b >= uv1.y && vs_b <= uv2.y) || (vs_b <= uv1.y && vs_b >= uv2.y)) ){
            uvs = make_vec2(us_b, vs_b);  
            // std::cout << "uvs = " << uvs << std::endl;
            
        }else{
            std::cerr << "ERROR (interpolate_texture_UV_to_slice_point): could not interpolate uv coordinates" << std::endl;
            throw(1);
        }
    }
    
    
    return uvs;  
    
}
