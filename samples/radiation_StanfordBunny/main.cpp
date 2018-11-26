#include "Context.h"
#include "Visualizer.h"
#include "RadiationModel.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  //RadiationModel radiationmodel(&context);

  std::vector<uint> bunny;

  //bunny = context.loadPLY("../../../PLY/StanfordBunny.ply",make_vec3(0,0,0),4.f);
  
  uint UUID;
  // float D = 10;
  // int2 size(5,5);
  // int2 subsize(50,50);
  // vec2 dx(D/float(size.x*subsize.x),D/float(size.y*subsize.y));
  // float rho;
  // for( int j=0; j<size.y; j++ ){
  //   for( int i=0; i<size.x; i++ ){

  //     float rot = ((j*size.x+i)%3)*M_PI*0.5;

  //     for( uint jj=0; jj<subsize.y; jj++ ){
  // 	for( uint ii=0; ii<subsize.x; ii++ ){

  // 	  if( (j*size.x+i)%2==0 ){
  // 	    UUID = context.addPatch( make_vec3(-0.5*D+(i*subsize.x+ii)*dx.x,-0.5*D+(j*subsize.y+jj)*dx.y,0), dx, make_SphericalCoord(0.f,rot), RGB::silver );
  // 	    rho = 0.f;
  // 	    context.setPrimitiveData(UUID,"reflectivity_SW",HELIOS_TYPE_FLOAT,1,&rho);
  // 	  }else{
  // 	    UUID = context.addPatch( make_vec3(-0.5*D+(i*subsize.x+ii)*dx.x,-0.5*D+(j*subsize.y+jj)*dx.y,0), dx, make_SphericalCoord(0.f,rot), RGB::white );
  // 	    rho = 0.6f;
  // 	    context.setPrimitiveData(UUID,"reflectivity_SW",HELIOS_TYPE_FLOAT,1,&rho);
  // 	  }

  // 	  context.setPrimitiveData( UUID, "radiation_flux_SW", 500.f );

  // 	}
  //     }

  //   }
  // }

  UUID=context.addPatch( make_vec3(0,0,8), make_vec2(1,1), make_SphericalCoord(0,0), "plugins/visualizer/textures/AlmondLeaf.png" );

  context.setPrimitiveData( UUID, "radiation_flux_SW", 700.f );
  //context.setPrimitiveData( bunny, "radiation_flux_SW", 600.f );
  
  //-----------------//

  vec3 sun_dir(0.4,-0.4,0.6);

  //RadiationModel radiationmodel(&context);

  // uint SunSource = radiationmodel.addCollimatedRadiationSource( sun_dir );
  
  // radiationmodel.addRadiationBand("SW");
  // radiationmodel.disableEmission("SW");
  // radiationmodel.setDirectRayCount("SW",100);
  // radiationmodel.setDiffuseRayCount("SW",300);
  // radiationmodel.setSourceFlux(SunSource,"SW",800.f);
  // radiationmodel.setDiffuseRadiationFlux("SW",200.f);
  // radiationmodel.setScatteringDepth("SW",3);
  
  // radiationmodel.updateGeometry();

  // radiationmodel.runBand("SW");

  Visualizer vis( 900 );

  vis.setLightingModel( Visualizer::LIGHTING_NONE );

  vis.buildContextGeometry(&context);

  //vis.colorContextPrimitivesByData( "radiation_flux_SW" );

  vis.enableColorbar();
  vis.setColorbarRange(400,1000);
  vis.setColorbarTitle("Radiation Flux");

  vis.setCameraPosition( make_SphericalCoord(11,0.4,2.5), make_vec3(0,0,0) );
  
  vis.plotInteractive();

}
