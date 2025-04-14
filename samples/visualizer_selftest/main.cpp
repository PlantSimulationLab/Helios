#include "Visualizer.h"

using namespace helios;

int main(){

    Context context;
    
    context.loadOBJ("/Users/bnbailey/Downloads/canopy.obj");
    
    Visualizer visualizer(800);
    
    visualizer.buildContextGeometry(&context);
    
    visualizer.plotInteractive();
    
  //Visualizer visualizer(100);
  
  //Run the self-test
  //return visualizer.selfTest();
	
}
