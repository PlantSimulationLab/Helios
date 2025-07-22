#include "RadiationModel.h"

using namespace helios;

int main() {

    Context context;

    RadiationModel radiationmodel(&context);

    return radiationmodel.selfTest();
}
