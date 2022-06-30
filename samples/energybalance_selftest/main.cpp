#include "EnergyBalanceModel.h"

using namespace helios;

int main() {
    Context context;

    EnergyBalanceModel energymodel(&context);

    return energymodel.selfTest();
}
