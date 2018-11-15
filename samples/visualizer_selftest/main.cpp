#include "Visualizer.h"

using namespace helios;

int main(void){

  Visualizer visualizer(100);
  
  //Run the self-test
  return visualizer.selfTest();
	
}
