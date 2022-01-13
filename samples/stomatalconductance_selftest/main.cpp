#include "StomatalConductanceModel.h"

using namespace helios;

int main(){
  
  Context context;

  StomatalConductanceModel stomatalconductance(&context);

  return stomatalconductance.selfTest();

}
