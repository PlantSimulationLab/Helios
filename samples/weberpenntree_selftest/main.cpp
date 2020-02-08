#include "Context.h"
#include "WeberPennTree.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  WeberPennTree wpt(&context);

  return wpt.selfTest();

}
