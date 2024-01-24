/** \file "selfTest.cpp" Self-test routines for LiDAR plug-in.

Copyright (C) 2016-2024 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "LiDAR.h"

using namespace std;
using namespace helios;

int LiDARcloud::selfTest() {

  std::cout << "Running LiDAR self-test..." << std::endl;;

  int fail_flag = 0;

  float LAD, LAD_exact;

  vec3 gsize;

  //------ Sphere Inside One Voxel ------//

  std::cout << "Running single voxel sphere test..." << std::flush;

  LiDARcloud pointcloud;
  pointcloud.disableMessages();

  pointcloud.loadXML("plugins/lidar/xml/sphere.xml");

  pointcloud.triangulateHitPoints( 0.5, 5 );

  Context context_1;

  pointcloud.addTrianglesToContext( &context_1 );

  if( context_1.getPrimitiveCount() != 386 ){
    std::cout << "failed." << std::endl;
    fail_flag++;
  }else{
    std::cout << "passed." << std::endl;
  }

  //------ Isotropic Patches Inside One Voxel ------//

  std::cout << "Running single voxel isotropic patches test..." << std::flush;

  LiDARcloud synthetic_1;
  synthetic_1.disableMessages();

  synthetic_1.loadXML(  "plugins/lidar/xml/synthetic_test.xml" );

  gsize = synthetic_1.getCellSize(0);

  Context context_2;

  std::vector<uint> UUIDs_1 = context_2.loadXML("plugins/lidar/xml/leaf_cube_LAI2_lw0_01_spherical.xml", true );

  LAD_exact = 0.f;
  for( uint UUID : UUIDs_1 ){
    LAD_exact += context_2.getPrimitiveArea(UUID)/(gsize.x*gsize.y*gsize.z);
  }

  synthetic_1.syntheticScan( &context_2 );

  synthetic_1.triangulateHitPoints( 0.04, 10 );
  synthetic_1.calculateLeafAreaGPU();

  LAD = synthetic_1.getCellLeafAreaDensity( 0 );

  if( fabs(LAD-LAD_exact)/LAD_exact>0.02 || LAD!=LAD ){
    std::cout << "failed." << std::endl;
    std::cout << "LAD: " << LAD << " " << LAD_exact << std::endl;
    fail_flag++;
  }else{
    std::cout << "passed." << std::endl;
  }

  //------ Isotropic Patches Inside Eight Voxels ------//

  std::cout << "Running eight voxel isotropic patches test..." << std::flush;

  LiDARcloud synthetic_2;
  synthetic_2.disableMessages();

  synthetic_2.loadXML(  "plugins/lidar/xml/synthetic_test_8.xml" );

  gsize = synthetic_2.getCellSize(0);

  std::vector<float> LAD_ex(8,0);
  for( uint UUID : UUIDs_1 ){
    int i,j,k;
    i = j = k = 0;
    vec3 v = context_2.getPrimitiveVertices(UUID).front();
    if( v.x>0.f ) {
      i = 1;
    }
    if( v.y>0.f ) {
      j = 1;
    }
    if( v.z>0.5f ) {
      k = 1;
    }
    int ID = k*4+j*2+i;

    float area = context_2.getPrimitiveArea(UUID);
    LAD_ex.at(ID) += area/(gsize.x*gsize.y*gsize.z);
  }

  synthetic_2.syntheticScan( &context_2);

  synthetic_2.triangulateHitPoints( 0.04, 10 );
  synthetic_2.calculateLeafAreaGPU();

  float RMSE = 0.f;
  for( int i=0; i<synthetic_2.getGridCellCount(); i++ ){

    LAD = synthetic_2.getCellLeafAreaDensity( i );
    RMSE += powf(LAD-LAD_ex.at(i),2)/float(synthetic_2.getGridCellCount());

  }
  RMSE = sqrtf(RMSE);

  if( RMSE>0.05 ){
    std::cout << "failed." << std::endl;
    std::cout << "RMSE: " << RMSE << std::endl;
    fail_flag++;
  }else{
    std::cout << "passed." << std::endl;
  }

  //------ Anisotropic Patches Inside One Voxel ------//

  std::cout << "Running single voxel anisotropic patches test..." << std::flush;

  LiDARcloud synthetic_3;
  synthetic_3.disableMessages();

  synthetic_3.loadXML(  "plugins/lidar/xml/synthetic_test.xml" );

  gsize = synthetic_3.getCellSize(0);

  LAD_exact = 0.f;
  for( uint UUID : UUIDs_1 ){
    LAD_exact += context_2.getPrimitiveArea(UUID)/(gsize.x*gsize.y*gsize.z);
  }

  synthetic_3.syntheticScan( &context_2 );

  synthetic_3.triangulateHitPoints( 0.04, 10 );
  synthetic_3.calculateLeafAreaGPU();

  LAD = synthetic_3.getCellLeafAreaDensity( 0 );

  if( fabs(LAD-LAD_exact)/LAD_exact>0.03 || LAD!=LAD ){
    std::cout << "failed." << std::endl;
    std::cout << "LAD: " << LAD << " " << LAD_exact << std::endl;
    fail_flag++;
  }else{
    std::cout << "passed." << std::endl;
  }

  //------ Synthetic Scan of Almond Tree ------//

  std::cout << "Running synthetic almond tree test..." << std::flush;

  Context context_4;

  context_4.loadOBJ("plugins/lidar/xml/AlmondWP.obj",make_vec3(0,0,0),6.,make_SphericalCoord(0,0),RGB::red,true);

  LiDARcloud synthetic_4;
  synthetic_4.disableMessages();

  synthetic_4.loadXML( "plugins/lidar/xml/almond.xml" );

  synthetic_4.syntheticScan( &context_4 );

  synthetic_4.calculateSyntheticLeafArea( &context_4 );
  synthetic_4.calculateSyntheticGtheta( &context_4 );

  synthetic_4.triangulateHitPoints( 0.05, 5 );
  synthetic_4.calculateLeafAreaGPU();

  //calculate exact leaf area

  uint Ncells = synthetic_4.getGridCellCount();

  std::vector<float> total_area;
  total_area.resize(Ncells);

  std::vector<float> Gtheta;
  Gtheta.resize(Ncells);

  std::vector<float> area_sum;
  area_sum.resize(Ncells,0.f);
  std::vector<float> sin_sum;
  sin_sum.resize(Ncells,0.f);
  std::vector<uint> cell_tri_count;
  cell_tri_count.resize(Ncells,0);

  std::vector<uint> UUIDs = context_4.getAllUUIDs();
  for( int p=0; p<UUIDs.size(); p++ ){

    uint UUID = UUIDs.at(p);

    if( context_4.doesPrimitiveDataExist(UUID,"gridCell") ){

      uint gridCell;
      context_4.getPrimitiveData(UUID, "gridCell",gridCell);

      if( gridCell>=0 && gridCell<Ncells ){
        total_area.at(gridCell) += context_4.getPrimitiveArea(UUID);
      }

      for( int s=0; s<synthetic_4.getScanCount(); s++ ){
        vec3 origin = synthetic_4.getScanOrigin(s);
        std::vector<vec3> vertices = context_4.getPrimitiveVertices(p);
        float area = context_4.getPrimitiveArea(p);
        vec3 normal = context_4.getPrimitiveNormal(p);
        vec3 raydir = vertices.front()-origin;
        raydir.normalize();
        float theta = fabs(acos_safe(raydir.z));

        if( area==area ){ //in rare cases you can get area=NaN

          Gtheta.at(gridCell) += fabs(normal*raydir)*area*fabs(sin(theta));

          area_sum.at(gridCell) += area;
          sin_sum.at(gridCell) += fabs(sin(theta));
          cell_tri_count.at(gridCell) += 1;

        }
      }

    }
  }

  for( uint v=0; v<Ncells; v++ ){
    if( cell_tri_count[v]>0 ){
      Gtheta[v] *= float(cell_tri_count[v])/(area_sum[v]*sin_sum[v]);
    }
  }

  float RMSE_LAD = 0.f;
  float bias_LAD = 0.f;
  float RMSE_Gtheta = 0.f;
  for( uint i=0; i<Ncells; i++ ){
    LAD = synthetic_4.getCellLeafArea(i);
    if( LAD==LAD && total_area.at(i)>0 && total_area.at(i)==total_area.at(i) ){
      RMSE_LAD += pow( LAD - total_area.at(i), 2)/float(Ncells);
      bias_LAD += (LAD - total_area.at(i))/float(Ncells);
    }
    float Gtheta_bar = synthetic_4.getCellGtheta(i);
    if( Gtheta_bar==Gtheta_bar && Gtheta.at(i)>0 && Gtheta.at(i)==Gtheta.at(i) ){
      RMSE_Gtheta += pow( Gtheta_bar - Gtheta.at(i), 2)/float(Ncells);
    }
  }
  RMSE_LAD = sqrt(RMSE_LAD);
  RMSE_Gtheta = sqrt(RMSE_Gtheta);

  if( RMSE_LAD>0.25 || bias_LAD>0 || RMSE_Gtheta>0.15 || RMSE_LAD==0.f ){
    std::cout << "failed." << std::endl;
    std::cout << "RMSE_LAD: " << RMSE_LAD << std::endl;
    std::cout << "bias_LAD: " << bias_LAD << std::endl;
    std::cout << "RMSE_Gtheta: " << RMSE_Gtheta << std::endl;
    fail_flag++;
  }else{
    std::cout << "passed." << std::endl;
  }

  //synthetic_4.exportLeafAreas("../output/leaf_areas.txt");
  //synthetic_4.exportGtheta("../output/Gtheta.txt");

  if( fail_flag==0 ){
    std::cout << "Passed all tests." << std::endl;
    return 0;
  }else{
    std::cout << "Failed " << fail_flag << " tests." << std::endl;
    return 1;
  }

}