/*
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#ifndef HELIOS_BUILDGEOMETRY_H
#define HELIOS_BUILDGEOMETRY_H

// Forward Declaration
class PlantArchitecture;

#include "Context.h"
#include "PlantArchitecture.h"

void BuildGeometry(const std::string &xml_input_file, PlantArchitecture *plant_architecture_ptr, helios::Context *context_ptr, std::vector<std::vector<uint>> &canopy_UUID_vector, std::vector<std::vector<helios::vec3>> &individual_plant_locations);

#endif // HELIOS_BUILDGEOMETRY_H
