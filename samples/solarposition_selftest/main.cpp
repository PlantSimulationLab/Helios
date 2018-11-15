#include "Context.h"
#include "SolarPosition.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  SolarPosition solarposition(&context);

  return solarposition.selfTest();

}
