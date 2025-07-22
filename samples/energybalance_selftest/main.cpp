#include "EnergyBalanceModel.h"

using namespace helios;

int main() {

    Context context;

    EnergyBalanceModel energybalance(&context);

    return energybalance.selfTest();
}
