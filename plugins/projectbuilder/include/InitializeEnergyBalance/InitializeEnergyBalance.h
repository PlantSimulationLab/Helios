#ifndef HELIOS_INITIALIZEENERGYBALANCE_H
#define HELIOS_INITIALIZEENERGYBALANCE_H

// Forward Declaration
class BLConductanceModel;
class EnergyBalanceModel;

#include "BoundaryLayerConductanceModel.h"
#include "EnergyBalanceModel.h"

void InitializeEnergyBalance(const std::string &xml_input_file, BLConductanceModel *boundarylayerconductancemodel, EnergyBalanceModel *energybalancemodel, helios::Context *context_ptr);

#endif // HELIOS_INITIALIZEENERGYBALANCE_H
