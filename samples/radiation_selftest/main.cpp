#include "RadiationModel.h"

using namespace helios;

int main()
{
  
  Context context;

  uint UUID = context.addPatch();

  RadiationModel radiationmodel(&context);

//  context.loadXML("plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml");

    uint SourceID = radiationmodel.addSunSphereRadiationSource(); //add a radiation source

    radiationmodel.setSourceSpectrum(SourceID, "solar_spectrum_ASTMG173"); //set the source flux spectral distribution to ASTMG173 solar spectrum
    radiationmodel.addRadiationBand("PAR", 400, 700); //add a radiation band called "PAR" defined between 400-700nm
//    radiationmodel.setSourceSpectrumIntegral(SourceID, 300, 400, 700 ); //set the integral of solar flux spectrum from 400-700nm to 300 W/m^2
    //radiationmodel.setSourceFlux( SourceID, "PAR", 300 );

    radiationmodel.updateGeometry();

    radiationmodel.runBand("PAR");

    float PAR;
    context.getPrimitiveData(UUID,"radiation_flux_PAR",PAR);

    std::cout << "PAR is " << PAR << std::endl;

  //return radiationmodel.selfTest();
    return 0;

}
