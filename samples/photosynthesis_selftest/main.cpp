#include "Context.h"
#include "PhotosynthesisModel.h"

using namespace helios;

int main(int argc, char* argv[])
{
  
  Context context;

  PhotosynthesisModel photosynthesis(&context);

  return photosynthesis.selfTest();

}
