#include "SolarPosition.h"

using namespace helios;

int main(){
  
  Context context;

  SolarPosition solarposition(&context);

  return solarposition.selfTest();

}
