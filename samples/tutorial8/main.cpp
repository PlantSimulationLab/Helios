#include "Visualizer.h"

using namespace helios; //note that we are using the helios namespace so we can omit 'helios::' before names
     
int main(){

  Context context;   //Declare the "Context" class

  // Load 3D model "Stanford Bunny" with a base position of (0,0,0) and scaled to a height of 2
  std::vector<uint> UUIDs = context.loadPLY( "../../../PLY/StanfordBunny.ply", make_vec3(0,0,0), 2, nullrotation, RGB::black );

  // Assign primitive data called "height" which gives the primitive z-coordinate
  for( uint UUID : UUIDs ){ //looping over primitive UUIDs
    std::vector<vec3> vertices = context.getPrimitiveVertices(UUID);  //get a vector containing the (x,y,z) position of each primitive vertex
    vec3 vertex = vertices.at(0);  //get the first vertex
    float z = vertex.z;   //get the vertex z-coordinate
    context.setPrimitiveData(UUID,"height",z);  //set this primitive's primitive data "height" equal to the value of "z"
  }

  // Visualize the result
  Visualizer visualizer(800);

  visualizer.buildContextGeometry(&context);

  visualizer.colorContextPrimitivesByData( "height" ); //color primitives based on a pseudocolor mapping of primitive data "height"

  visualizer.setColormap( Visualizer::COLORMAP_PARULA );  //change the colormap to "parula"
  visualizer.setColorbarTitle("Height" ); //give our colorbar a title

  visualizer.plotInteractive();

  return 0;

}
