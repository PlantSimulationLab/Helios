#ifndef HELIOS_BUILDGEOMETRY_H
#define HELIOS_BUILDGEOMETRY_H

// Forward Declaration
class PlantArchitecture;

#include "Context.h"
#include "PlantArchitecture.h"

void BuildGeometry(const std::string &xml_input_file, PlantArchitecture *plant_architecture_ptr, helios::Context *context_ptr, std::vector<std::vector<uint>> &canopy_UUID_vector,  std::vector<std::vector<helios::vec3>> &individual_plant_locations);

#endif //HELIOS_BUILDGEOMETRY_H
