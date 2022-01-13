#include "Context.h"

int main(){

  //Declare and initialize the Helios context
  helios::Context context;

  //Run the self-test
  return context.selfTest();

}
