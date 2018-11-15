#include "Context.h"

using namespace helios;

int main(void){

  //Declare and initialize the Helios context
  //note that since we have used the `helios' namespace above, we do not need to declare the context as: helios::Context
  Context context;

  //Run the self-test
  return context.selfTest();
	
}
