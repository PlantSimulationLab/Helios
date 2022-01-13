#include "Visualizer.h"
#include "RadiationModel.h"
#include "EnergyBalanceModel.h"

using namespace helios;

int main(){
  
  Context context;

  //---- CREATING GEOMETRY ----- //

  std::vector<uint> bunny;

  bunny = context.loadPLY("../../../PLY/StanfordBunny.ply",make_vec3(0,0,0),4.f);
  
  uint UUID;
  float D = 10;
  int2 size(5,5);
  int2 subsize(50,50);
  vec2 dx(D/float(size.x*subsize.x),D/float(size.y*subsize.y));
  float rho;
  for( int j=0; j<size.y; j++ ){
    for( int i=0; i<size.x; i++ ){

      float rot = ((j*size.x+i)%3)*M_PI*0.5;

      for( uint jj=0; jj<subsize.y; jj++ ){
  	for( uint ii=0; ii<subsize.x; ii++ ){

  	  if( (j*size.x+i)%2==0 ){
  	    UUID = context.addPatch( make_vec3(-0.5*D+(i*subsize.x+ii)*dx.x,-0.5*D+(j*subsize.y+jj)*dx.y,0), dx, make_SphericalCoord(0.f,rot), RGB::silver );
  	    rho = 0.f;
  	    context.setPrimitiveData( UUID, "reflectivity_SW",HELIOS_TYPE_FLOAT,1,&rho);
  	  }else{
  	    UUID = context.addPatch( make_vec3(-0.5*D+(i*subsize.x+ii)*dx.x,-0.5*D+(j*subsize.y+jj)*dx.y,0), dx, make_SphericalCoord(0.f,rot), RGB::white );
  	    rho = 0.6f;
  	    context.setPrimitiveData( UUID, "reflectivity_SW",HELIOS_TYPE_FLOAT,1,&rho);
  	  }

  	}
      }

    }
  }

  //---- SET UP RADIATION MODEL ----- //
    
  vec3 sun_dir(0.4,-0.4,0.6);

  RadiationModel radiationmodel(&context);

  uint SunSource = radiationmodel.addCollimatedRadiationSource( sun_dir );
  
  radiationmodel.addRadiationBand("SW");
  radiationmodel.disableEmission("SW");
  radiationmodel.setDirectRayCount("SW",100);
  radiationmodel.setDiffuseRayCount("SW",1000);
  radiationmodel.setSourceFlux(SunSource,"SW",600.f);
  radiationmodel.setDiffuseRadiationFlux("SW",100);
  radiationmodel.setScatteringDepth("SW",3);

  radiationmodel.addRadiationBand("LW");
  radiationmodel.setDiffuseRayCount("LW",1000);
  radiationmodel.setDiffuseRadiationFlux("LW",5.67e-8*pow(300,4));
  
  radiationmodel.updateGeometry();

  radiationmodel.runBand("SW");
  radiationmodel.runBand("LW");

  //---- SET UP ENERGY BALANCE MODEL ----- //

  EnergyBalanceModel energybalancemodel(&context);

  energybalancemodel.addRadiationBand("SW");
  energybalancemodel.addRadiationBand("LW");
  
  energybalancemodel.run();

  //---- SET UP RADIATION MODEL ----- //

  Visualizer vis( 1200 );

  vis.setLightingModel( Visualizer::LIGHTING_NONE );

  vis.setCameraPosition(make_SphericalCoord(15,0.35,0.4*M_PI), make_vec3(0,0,2) );

  vis.buildContextGeometry(&context);

  vis.colorContextPrimitivesByData( "temperature" );

  vis.enableColorbar();
  vis.setColorbarRange(300,320);
  vis.setColorbarTitle("Temperature (K)");
  
  vis.plotInteractive();

}
