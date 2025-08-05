#include "StomatalConductanceModel.h"

using namespace helios;

int main(int argc, char** argv) {
    Context context;
    StomatalConductanceModel gs(&context);
    return gs.selfTest(argc, argv);
}
