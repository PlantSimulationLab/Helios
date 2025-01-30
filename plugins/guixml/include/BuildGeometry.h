#ifndef HELIOS_BUILDGEOMETRY_H
#define HELIOS_BUILDGEOMETRY_H

#include "PlantArchitecture.h"
#include "Context.h"

void BuildGeometry(const std::string &xml_input_file, PlantArchitecture *plant_architecture_ptr, helios::Context *context_ptr);

#endif //HELIOS_BUILDGEOMETRY_H
