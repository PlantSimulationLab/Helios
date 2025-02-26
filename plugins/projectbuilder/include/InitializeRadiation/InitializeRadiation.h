#ifndef HELIOS_INITIALIZERADIATION_H
#define HELIOS_INITIALIZERADIATION_H

// Forward Declaration
class RadiationModel;
class SolarPosition;

#include "RadiationModel.h"
#include "SolarPosition.h"

void InitializeRadiation(const std::string &xml_input_file, SolarPosition *solarposition_ptr, RadiationModel *radiation_ptr, helios::Context *context_ptr );

#endif //HELIOS_INITIALIZERADIATION_H
