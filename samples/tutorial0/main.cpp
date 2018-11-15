#include "Context.h"

int main(void){

  //Declare and initialize the Helios context
  helios::Context context;

  //Run the self-test
  context.selfTest();

  return 0;

}
