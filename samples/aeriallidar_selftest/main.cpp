#include "Context.h"
#include "AerialLiDAR.h"

using namespace helios;

int main( void ){

  AerialLiDARcloud aeriallidar;

  return aeriallidar.selfTest();

}
