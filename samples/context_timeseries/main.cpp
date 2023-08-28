#include "Context.h"
#include "Visualizer.h" //include this header to use Visualizer

using namespace helios;

int main(){

  //Declare and initialize the helios context
  //note that since we have used the `helios' namespace above, we do not need to declare the context as: helios::Context
  Context context;

  std::vector<uint> bunny;
  std::vector<uint> ground;
  
  bunny = context.loadPLY("../../../PLY/StanfordBunny.ply",make_vec3(0,0,0),4.f);

  float D = 50;
  int2 size(15,15);
  vec2 dx(D/float(size.x),D/float(size.y));
  float rho;
  for( int j=0; j<size.y; j++ ){
    for( int i=0; i<size.x; i++ ){

      float rot = ((j*size.x+i)%3)*M_PI*0.5;

      if( (j*size.x+i)%2==0 ){
  	ground.push_back( context.addPatch( make_vec3(-0.5*D+i*dx.x,-0.5*D+j*dx.y,0), dx, make_SphericalCoord(0.f,rot), RGB::silver ) );
      }else{
  	ground.push_back( context.addPatch( make_vec3(-0.5*D+i*dx.x,-0.5*D+j*dx.y,0), dx, make_SphericalCoord(0.f,rot), RGB::white ) );
      }

    }
  }

  //Load the XML file with temperature data, which will create a timeseries named "temperature"
  context.loadXML( "../xml/temperature.xml" );

  //Number to data points in timeseries
  uint N = context.getTimeseriesLength("temperature");

  //Initialize the visualizer
  Visualizer vis( 1000 );
  
  vis.setBackgroundColor( make_RGBcolor( 0.8, 0.8, 1 ) );

  vis.setColormap(Visualizer::COLORMAP_HOT);
  vis.setColorbarRange( 20, 40 );

  vis.setLightingModel( Visualizer::LIGHTING_PHONG );
  vis.setLightDirection( make_vec3(1,1,1) );

  vis.buildContextGeometry(&context);
  
  //loop over timeseries data
  for( uint t=0; t<N; t++ ){
    
    context.setCurrentTimeseriesPoint( "temperature", t );

    float T = context.queryTimeseriesData( "temperature", t );

    for( uint j=0; j<bunny.size(); j++ ){
      context.setPrimitiveData( bunny.at(j), "temperature", T );
    }

    vis.buildContextGeometry(&context);

    vis.colorContextPrimitivesByData( "temperature", bunny );
    
    vis.setCameraPosition( make_SphericalCoord(10,0.05*M_PI,0.2f*M_PI), make_vec3(0,0,1) );
  
    vis.plotUpdate();
    
    helios::wait(0.01);

    vis.clearGeometry();
    
  }

}
