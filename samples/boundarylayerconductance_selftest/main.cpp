#include "BoundaryLayerConductanceModel.h"

using namespace helios;

int main(int argc, char** argv) {
    Context context;
    BLConductanceModel blc(&context);
    return blc.selfTest(argc, argv);
}
