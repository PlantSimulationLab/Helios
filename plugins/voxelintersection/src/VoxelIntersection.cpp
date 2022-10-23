/** \file "VoxelIntersection.cpp" Primary source file for Primitive-Voxel Intersection plug-in.
    \author Brian Bailey

    Copyright (C) 2016-2022 Brian Bailey, Eric Kent

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

    if(voxel_face_vertices.size() < 3)
    {
        std::cerr << "ERROR (slicePrimitive): voxel_face_verticies must contain at least three points" << std::endl;
        throw(1);
    }

    helios::vec3 face_normal = cross(voxel_face_vertices.at(1) - voxel_face_vertices.at(0), voxel_face_vertices.at(2) - voxel_face_vertices.at(1));
    face_normal.normalize();

    std::vector<helios::vec3> patch_vertices = context->getPrimitiveVertices(UUID);
    helios::vec3 patch_normal = context->getPrimitiveNormal(UUID);
    patch_normal.normalize();

    //BNB
    helios::RGBAcolor patch_color = context->getPrimitiveColorRGBA(UUID);

    std::string texa;
    const char * tex;
    texa =  context->getPrimitiveTextureFile(UUID);
    tex = texa.c_str();
    bool patchHasTexture = !texa.empty();

    //////////////////////////////////////////////////////////////////////////////////////////////////////////
    // find the equation of the line where the planes of the patch and voxel face intersect

    // direction of the plane intersection line
    helios::vec3 direction_vector = cross(face_normal, patch_normal);

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
    helios::vec3 d2a = helios::make_vec3(-1*patch_normal.x*patch_vertices.at(1).x, -1*patch_normal.y*patch_vertices.at(1).y, -1*patch_normal.z*patch_vertices.at(1).z);

    float d1 = d1a.x + d1a.y + d1a.z;
    float d2 = d2a.x + d2a.y + d2a.z;

    float xi;
    float yi;
    float zi;

    if(maxc == 1)
    {
        xi = 0;
        yi  = (d2*face_normal.z - d1*patch_normal.z) /  direction_vector.x;
        zi = (d1*patch_normal.y - d2*face_normal.y) /  direction_vector.x;
    }else if(maxc == 2){
        xi = (d1*patch_normal.z - d2*face_normal.z) /  direction_vector.y;
        yi = 0;
        zi = (d2*face_normal.x - d1*patch_normal.x) /  direction_vector.y;
    }else if(maxc == 3){
        xi = (d2*face_normal.y - d1*patch_normal.y) /  direction_vector.z;
        yi = (d1*patch_normal.x - d2*face_normal.x) /  direction_vector.z;
        zi = 0 ;
    }

    helios::vec3  ipoint = make_vec3(xi, yi, zi);

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // get points of intersection between each edge of the patch and the patch-voxel intersection line

    // vector for points of intersection between edge line and intersection line
    std::vector<helios::vec3> possible_points;
    // vector for points that actually touch the patch
    std::vector<helios::vec3> actual_points;
    std::vector<uint> actual_points_edge_ID;

    helios::vec3 vi0;
    helios::vec3 vi1;

    // go through the different edges of the patch and calculate intersection points with line along edge of patch and intersection line
    if(patch_vertices.size() == 4)
    {
        possible_points.resize(4);
        possible_points.at(0) = linesIntersection(patch_vertices.at(1), patch_vertices.at(1) - patch_vertices.at(0), ipoint, direction_vector);
        possible_points.at(1) = linesIntersection(patch_vertices.at(2), patch_vertices.at(2) - patch_vertices.at(1), ipoint, direction_vector);
        possible_points.at(2) = linesIntersection(patch_vertices.at(3), patch_vertices.at(3) - patch_vertices.at(2), ipoint, direction_vector);
        possible_points.at(3) = linesIntersection(patch_vertices.at(0), patch_vertices.at(0) - patch_vertices.at(3), ipoint, direction_vector);

        for(uint i=0;i<4;i++)
        {
            if(i == 0)
            {
                vi1 = patch_vertices.at(1);
                vi0 = patch_vertices.at(0);
            }else if(i == 1){
                vi1 = patch_vertices.at(2);
                vi0 = patch_vertices.at(1);
            }else if(i == 2){
                vi1 = patch_vertices.at(3);
                vi0 = patch_vertices.at(2);
            }else if(i == 3){
                vi1 = patch_vertices.at(0);
                vi0 = patch_vertices.at(3);
            }

            bool test_x = ( (possible_points.at(i).x >= vi1.x && possible_points.at(i).x <= vi0.x) || (possible_points.at(i).x >= vi0.x && possible_points.at(i).x <= vi1.x) );
            bool test_y = ( (possible_points.at(i).y >= vi1.y && possible_points.at(i).y <= vi0.y) || (possible_points.at(i).y >= vi0.y && possible_points.at(i).y <= vi1.y) );
            bool test_z = ( (possible_points.at(i).z >= vi1.z && possible_points.at(i).z <= vi0.z) || (possible_points.at(i).z >= vi0.z && possible_points.at(i).z <= vi1.z) );

            if( test_x && test_y && test_z){
                actual_points.push_back(possible_points.at(i));
                actual_points_edge_ID.push_back(i);
            }
        }

    }else if(patch_vertices.size() == 3){

        possible_points.resize(3);
        possible_points.at(0) = linesIntersection(patch_vertices.at(1), patch_vertices.at(1) - patch_vertices.at(0), ipoint, direction_vector);
        possible_points.at(1) = linesIntersection(patch_vertices.at(2), patch_vertices.at(2) - patch_vertices.at(1), ipoint, direction_vector);
        possible_points.at(2) = linesIntersection(patch_vertices.at(0), patch_vertices.at(0) - patch_vertices.at(2), ipoint, direction_vector);

        for(uint i=0;i<3;i++)
        {
            if(i == 0)
            {
                vi1 = patch_vertices.at(1);
                vi0 = patch_vertices.at(0);
            }else if(i == 1){
                vi1 = patch_vertices.at(2);
                vi0 = patch_vertices.at(1);

            }else if(i == 2){
                vi1 = patch_vertices.at(0);
                vi0 = patch_vertices.at(2);
            }

            bool test_x = ( (possible_points.at(i).x >= vi1.x && possible_points.at(i).x <= vi0.x) || (possible_points.at(i).x >= vi0.x && possible_points.at(i).x <= vi1.x) );
            bool test_y = ( (possible_points.at(i).y >= vi1.y && possible_points.at(i).y <= vi0.y) || (possible_points.at(i).y >= vi0.y && possible_points.at(i).y <= vi1.y) );
            bool test_z = ( (possible_points.at(i).z >= vi1.z && possible_points.at(i).z <= vi0.z) || (possible_points.at(i).z >= vi0.z && possible_points.at(i).z <= vi1.z) );

            if( test_x && test_y && test_z){
                actual_points.push_back(possible_points.at(i));
                actual_points_edge_ID.push_back(i);
            }
        }

    }

    //vector UUIDs that will be output
    std::vector<uint> resulting_UUIDs;

    // the primitive did not intersect with the voxel face
    if(actual_points.size() == 0){
        resulting_UUIDs.push_back(UUID);
        return resulting_UUIDs;
    }

    // the primitive intersected with the face at a single point (a corner) - no slicing needed
    if(actual_points.size() == 1){
        resulting_UUIDs.push_back(UUID);
        if(printmessages){
            std::cout << "the primitive intersected with the face at a single point (a corner) - no slicing needed" << std::endl;
        }
        return resulting_UUIDs;
    }

    // the primitive intersected with the face along an edge - no need to slice
    if(actual_points_edge_ID.at(0) == actual_points_edge_ID.at(1)){
        resulting_UUIDs.push_back(UUID);
        if(printmessages){
            std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
        }
        return resulting_UUIDs;
    }

    float absTol = pow(10, -6);
    float relTol = pow(10, -20);

    if(patch_vertices.size() == 4)
    {
        if((approxSame(actual_points.at(0), patch_vertices.at(0), absTol) && approxSame(actual_points.at(1), patch_vertices.at(1), absTol))  ||
           (approxSame(actual_points.at(0), patch_vertices.at(1), absTol) && approxSame(actual_points.at(1), patch_vertices.at(0), absTol)) ||
           (approxSame(actual_points.at(0), patch_vertices.at(1), absTol) && approxSame(actual_points.at(1), patch_vertices.at(2), absTol)) ||
           (approxSame(actual_points.at(0), patch_vertices.at(2), absTol) && approxSame(actual_points.at(1), patch_vertices.at(1), absTol)) ||
           (approxSame(actual_points.at(0), patch_vertices.at(2), absTol) && approxSame(actual_points.at(1), patch_vertices.at(3), absTol)) ||
           (approxSame(actual_points.at(0), patch_vertices.at(3), absTol) && approxSame(actual_points.at(1), patch_vertices.at(2), absTol)) ||
           (approxSame(actual_points.at(0), patch_vertices.at(3), absTol) && approxSame(actual_points.at(1), patch_vertices.at(0), absTol)) ||
           (approxSame(actual_points.at(0), patch_vertices.at(0), absTol) && approxSame(actual_points.at(1), patch_vertices.at(3), absTol)) )
        {
            if(printmessages){
                std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
            }
            resulting_UUIDs.push_back(UUID);
            return resulting_UUIDs;
        }

    }else if(patch_vertices.size() == 3){

        if((approxSame(actual_points.at(0), patch_vertices.at(0), absTol) || approxSame(actual_points.at(0), patch_vertices.at(1), absTol) ||
            approxSame(actual_points.at(0), patch_vertices.at(2), absTol)) &&
           (approxSame(actual_points.at(1), patch_vertices.at(0), absTol) || approxSame(actual_points.at(1), patch_vertices.at(1), absTol) ||
            approxSame(actual_points.at(1), patch_vertices.at(2), absTol))){
            resulting_UUIDs.push_back(UUID);
            if(printmessages){
                std::cout << "the primitive intersected with the face along an edge - no need to slice" << std::endl;
            }
            return resulting_UUIDs;
        }
    }

    // if the number of 'actual intersection points' is greater than 2, choose the points that are farthest apart from each other
    if(actual_points.size() == 3)
    {

        std::cout << "WARNING: 3 intersection points between primitive edge and voxel face for UUID " << UUID << ". Using points farthest apart." << std::endl;

        float d01 = point_distance(actual_points.at(0), actual_points.at(1));
        float d02 = point_distance(actual_points.at(0), actual_points.at(2));
        float d12 = point_distance(actual_points.at(1), actual_points.at(2));

        std::vector<vec3> temp_points;
        std::vector<uint> temp_edges;

        if(d01 > d02)
        {
            if(d01 > d12)
            {
                temp_points.push_back(actual_points.at(0));
                temp_points.push_back(actual_points.at(1));
                temp_edges.push_back(actual_points_edge_ID.at(0));
                temp_edges.push_back(actual_points_edge_ID.at(1));
            }else{
                temp_points.push_back(actual_points.at(1));
                temp_points.push_back(actual_points.at(2));
                temp_edges.push_back(actual_points_edge_ID.at(1));
                temp_edges.push_back(actual_points_edge_ID.at(2));
            }
        }else{
            if(d02 > d12)
            {
                temp_points.push_back(actual_points.at(0));
                temp_points.push_back(actual_points.at(2));
                temp_edges.push_back(actual_points_edge_ID.at(0));
                temp_edges.push_back(actual_points_edge_ID.at(2));
            }else{
                temp_points.push_back(actual_points.at(1));
                temp_points.push_back(actual_points.at(2));
                temp_edges.push_back(actual_points_edge_ID.at(1));
                temp_edges.push_back(actual_points_edge_ID.at(2));
            }
        }

        actual_points = temp_points;
        actual_points_edge_ID = temp_edges;


    }else if(actual_points.size() > 3){
        std::cout << "####################################################################" << std::endl;
        std::cout << "ERROR: actual_points.size() > 3" << std::endl;
        // haven't come across this case yet, didn't implement a fix
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
        if(patch_vertices.at(0).x < face_coordinate.x)
        {
            buffer = make_vec3(float(-1)*buffer_value, 0, 0);
        }else if(patch_vertices.at(0).x > face_coordinate.x){
            buffer = make_vec3(buffer_value, 0, 0);
        }else{
            if(printmessages){
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }

    }else if(fabs(face_normal.y) > 0.5){
        if(patch_vertices.at(0).y < face_coordinate.y )
        {
            buffer = make_vec3( 0,float(-1)*buffer_value, 0);
        }else if(patch_vertices.at(0).y > face_coordinate.y){
            buffer = make_vec3( 0,buffer_value, 0);
        }else{
            if(printmessages){
                std::cout << "vertex 0 exactly at face" << std::endl;
            }
        }

    }else if(fabs(face_normal.z) > 0.5){
        if(patch_vertices.at(0).z < face_coordinate.z )
        {
            buffer = make_vec3( 0, 0, float(-1)*buffer_value);
        }else if(patch_vertices.at(0).z > face_coordinate.z){
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

    /////////////////////////////////////////////////////////////////////////////////////////////
    // if the primitive isn't texture masked
    if(patchHasTexture == false){

        if(patch_vertices.size() == 4)
        {
            //cases where intersection points are on opposite sides
            if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 2))
            {
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 3)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 3 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, patch_color);
                //cases where intersection points are on adjacent sides
            }else if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 3)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 3 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(2) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(2) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 3 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(2) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 3)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, patch_color);
                t3 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(2) + buffer, patch_color);
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


        }else if( patch_vertices.size() == 3){

            if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 1))
            {
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(2) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(2) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(2) + buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(2) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
                t2 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);//
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);//
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_color);//
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);//
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(0) + buffer, patch_color);
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


    }else if(patchHasTexture){

        //////////////////////////////////////////////////////////////////////////////////////////////
        // rotate patch vertices and intersection points into the x-y plane for purposes of calculating uv coordinates
        helios::vec3 xy = make_vec3(0,0,1);
        helios::vec3 line_direction = cross(patch_normal, xy);
        std::vector<helios::vec3> rotated_actual_points_t = actual_points;
        std::vector<helios::vec3> rotated_patch_vertices_t= patch_vertices;

        if(line_direction == make_vec3(0,0,0))
        { // they are already in the xy plane
        }else{
            //rotate them
            float theta = acos_safe((patch_normal.z*xy.z)/(patch_normal.magnitude())*xy.magnitude());
            for(uint i=0;i<actual_points.size();i++)
            {
                rotated_actual_points_t.at(i) = rotatePointAboutLine(actual_points.at(i), patch_vertices.at(0), line_direction, theta );
            }

            for(uint i=0;i<patch_vertices.size();i++)
            {
                rotated_patch_vertices_t.at(i) = rotatePointAboutLine(patch_vertices.at(i), patch_vertices.at(0), line_direction, theta );
            }

        }

        //get uv coordinates of the vertices
        std::vector<helios::vec2> v_uv = context->getPrimitiveTextureUV(UUID);
        std::vector<helios::vec3> v_v = context->getPrimitiveVertices(UUID);

        //get uv coordinates of the intersection points
        std::vector<helios::vec2> ip_uv;
        ip_uv.resize(2);

        if(patch_vertices.size() == 4)
        {

            // it seems patches that are not explicitly set up with texture UV coordinates just don't have them
            // so set the default here
            if(v_uv.size() == 0)
            {
                std::vector<helios::vec2> uv{ make_vec2(0,0), make_vec2(1,0), make_vec2(1,1), make_vec2(0,1) };
                v_uv = uv;
            }

            // for each intersection point, choose the patch vertices on the corresponding edge
            for(uint i=0;i<rotated_actual_points_t.size();i++)
            {
                helios::vec3 point_0;
                helios::vec3 point_1;
                helios::vec2 point_uv;
                helios::vec2 point_0uv;
                helios::vec2 point_1uv;

                if(actual_points_edge_ID.at(i) == 0)
                {
                    point_0 = rotated_patch_vertices_t.at(0);
                    point_1 = rotated_patch_vertices_t.at(1);
                    point_0uv = v_uv.at(0);
                    point_1uv = v_uv.at(1);
                }else if(actual_points_edge_ID.at(i) == 1){
                    point_0 = rotated_patch_vertices_t.at(1);
                    point_1 = rotated_patch_vertices_t.at(2);
                    point_0uv = v_uv.at(1);
                    point_1uv = v_uv.at(2);
                }else if(actual_points_edge_ID.at(i) == 2){
                    point_0 = rotated_patch_vertices_t.at(2);
                    point_1 = rotated_patch_vertices_t.at(3);
                    point_0uv = v_uv.at(2);
                    point_1uv = v_uv.at(3);
                }else if(actual_points_edge_ID.at(i) == 3){
                    point_0 = rotated_patch_vertices_t.at(3);
                    point_1 = rotated_patch_vertices_t.at(0);
                    point_0uv = v_uv.at(3);
                    point_1uv = v_uv.at(0);
                }

                // check if the patch vertices for this edge are equal in x or y coordinate to prevent division by 0
                float absTol = pow(10, -4);
                float relTol = pow(10, -20);
                if(approxSame(point_1.x, point_0.x, absTol, relTol))
                {
                    point_uv.x =  point_0uv.x;
                }else{
                    point_uv.x =  point_0uv.x + (rotated_actual_points_t.at(i).x - point_0.x)/(point_1.x - point_0.x)*(point_1uv.x-point_0uv.x);
                }

                if(approxSame(point_1.y, point_0.y, absTol, relTol))
                {
                    point_uv.y =  point_0uv.y;
                }else{
                    point_uv.y =   point_0uv.y + (rotated_actual_points_t.at(i).y - point_0.y)/(point_1.y - point_0.y)*(point_1uv.y-point_0uv.y);
                }

                ip_uv.at(i) = point_uv;

                if(ip_uv.at(i).x < 0 || ip_uv.at(i).x > 1 || ip_uv.at(i).y < 0 || ip_uv.at(i).y > 1)
                {
                    std::cout << "ERROR: texture uv for UUID " << UUID << " < 0 or > 1" << std::endl;
                }

            }

        }else if(patch_vertices.size() == 3){


            for(uint i=0;i<rotated_actual_points_t.size();i++)
            {
                //vectors to hold point coordinates and uv coordinates for the points on the current point's edge
                helios::vec3 point_0;
                helios::vec3 point_1;

                helios::vec2 point_0uv;
                helios::vec2 point_1uv;
                helios::vec2 point_uv;

                if(actual_points_edge_ID.at(i) == 0)
                {
                    point_0 = rotated_patch_vertices_t.at(0);
                    point_1 = rotated_patch_vertices_t.at(1);
                    point_0uv = v_uv.at(0);
                    point_1uv = v_uv.at(1);
                }else if(actual_points_edge_ID.at(i) == 1){
                    point_0 = rotated_patch_vertices_t.at(1);
                    point_1 = rotated_patch_vertices_t.at(2);
                    point_0uv = v_uv.at(1);
                    point_1uv = v_uv.at(2);
                }else if(actual_points_edge_ID.at(i) == 2){
                    point_0 = rotated_patch_vertices_t.at(2);
                    point_1 = rotated_patch_vertices_t.at(0);
                    point_0uv = v_uv.at(2);
                    point_1uv = v_uv.at(0);
                }

                float absTol = pow(10, -4);
                float relTol = pow(10, -20);

                if(approxSame(point_1.x, point_0.x, absTol, relTol))
                {
                    point_uv.x =  point_0uv.x;
                }else{
                    point_uv.x =  point_0uv.x + (rotated_actual_points_t.at(i).x - point_0.x)/(point_1.x - point_0.x)*(point_1uv.x-point_0uv.x);
                }

                if(approxSame(point_1.y, point_0.y, absTol, relTol))
                {
                    point_uv.y =  point_0uv.y;
                }else{
                    point_uv.y =   point_0uv.y + (rotated_actual_points_t.at(i).y - point_0.y)/(point_1.y - point_0.y)*(point_1uv.y-point_0uv.y);
                }

                ip_uv.at(i) = point_uv;

                if(ip_uv.at(i).x < 0 || ip_uv.at(i).x > 1 || ip_uv.at(i).y < 0 || ip_uv.at(i).y > 1)
                {
                    std::cout << "ERROR: texture uv for UUID " << UUID << " < 0 or > 1" << std::endl;
                }

            }
        }

        if(patch_vertices.size() == 4)
        {
            //cases where intersection points are on opposite sides
            if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 2))
            {
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer,  patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(3));
                t3 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(3));
                t3 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 3)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
                t3 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2) ,v_uv.at(3));
            }else if((actual_points_edge_ID.at(0) == 3 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
                t3 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(1), v_uv.at(2) ,v_uv.at(3));
                //cases where intersection points are on adjacent sides
            }else if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 3)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
                t3 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(1), v_uv.at(2) ,v_uv.at(3));
            }else if((actual_points_edge_ID.at(0) == 3 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
                t3 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(2) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(0), v_uv.at(2) ,v_uv.at(3));
            }else if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(2) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(1), v_uv.at(2) ,v_uv.at(3));
                t3 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(3));
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(2) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(0), v_uv.at(2) ,v_uv.at(3));
                t3 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(3));
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(3));
                t3 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(3) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(3));
                t3 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
            }else if((actual_points_edge_ID.at(0) == 3 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(1));
                t3 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(2) + buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 3)){//checked
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(3) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(3));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(1) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(1));
                t3 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(2) + buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
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

        }else if(patch_vertices.size() == 3){

            //cases where intersection points are on opposite sides
            if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 1))
            {
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(2) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(2) + buffer, tex, ip_uv.at(0), v_uv.at(0) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(1));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(2) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, patch_vertices.at(2) + buffer, tex, ip_uv.at(1), v_uv.at(0) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 0 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(0) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 0)){
                t0 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(1) - buffer, patch_vertices.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(2));
            }else if((actual_points_edge_ID.at(0) == 1 && actual_points_edge_ID.at(1) == 2)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(0) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), v_uv.at(1) ,v_uv.at(0));
            }else if((actual_points_edge_ID.at(0) == 2 && actual_points_edge_ID.at(1) == 1)){
                t0 = context->addTriangle(actual_points.at(0) - buffer, actual_points.at(1) - buffer, patch_vertices.at(2) - buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(0));
                t1 = context->addTriangle(actual_points.at(0) + buffer, actual_points.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(0), ip_uv.at(1) ,v_uv.at(2));
                t2 = context->addTriangle(actual_points.at(1) + buffer, patch_vertices.at(1) + buffer, patch_vertices.at(0) + buffer, tex, ip_uv.at(1), v_uv.at(1) ,v_uv.at(0));
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
    }

    // copy over primitive data to the new triangles
    for(uint i=0;i<resulting_UUIDs.size();i++){
        context->copyPrimitiveData(UUID, resulting_UUIDs.at(i));
        //BNB
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

    // delete the original primitive
    // assert(resulting_UUIDs.size() > 0);

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
