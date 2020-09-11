#include "Context.h"
#include "CanopyGenerator.h"

using namespace helios;

int main(void){

  Context context;

  CanopyGenerator canopygenerator(&context);
  
  //Run the self-test
  return canopygenerator.selfTest();
	
}
