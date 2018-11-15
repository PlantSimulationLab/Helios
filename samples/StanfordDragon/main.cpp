#include "Context.h"
#include "Visualizer.h"

using namespace helios;

int main( void ){

  Context context;

  context.loadPLY("../../../PLY/StanfordDragon.ply", make_vec3(0,0,0), 1 );

  Visualizer vis(1200);

  vis.buildContextGeometry(&context);

  vis.setLightDirection( make_vec3(1,1,1) );

  vis.setLightingModel( Visualizer::LIGHTING_PHONG_SHADOWED );

  vis.plotInteractive();

}
