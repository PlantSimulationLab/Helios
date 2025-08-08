#include "EnergyBalanceModel.h"

using namespace helios;

int main(int argc, char** argv) {
    Context context;
    EnergyBalanceModel energybalance(&context);
    return energybalance.selfTest(argc, argv);
}
