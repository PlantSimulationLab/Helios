#include "BoundaryLayerConductanceModel.h"

using namespace helios;

int main() {
    Context context;

    BLConductanceModel blc(&context);

    blc.selfTest();
}
