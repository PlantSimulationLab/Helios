#include "RadiationModel.h"

using namespace helios;

int main(int argc, char **argv) {
    Context context;
    RadiationModel radiationmodel(&context);
    return radiationmodel.selfTest(argc, argv);
}
