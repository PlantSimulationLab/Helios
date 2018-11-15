#include "Context.h"
#include "EnergyBalanceModel.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  EnergyBalanceModel energymodel(&context);

  return energymodel.selfTest();

}
