#include "Context.h"
#include "StomatalConductanceModel.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  StomatalConductanceModel stomatalconductance(&context);

  return stomatalconductance.selfTest();

}
