#include "Context.h"
#include "BoundaryLayerConductanceModel.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  BLConductanceModel blc(&context);

  blc.selfTest();

}
