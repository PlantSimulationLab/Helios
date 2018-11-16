#include "Context.h"
#include "VoxelIntersection.h"

using namespace helios;

int main( void ){

  Context context;
  VoxelIntersection voxelintersection(&context);

  return voxelintersection.selfTest();
  
}
