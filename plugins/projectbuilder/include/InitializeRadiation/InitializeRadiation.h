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

#ifndef HELIOS_INITIALIZERADIATION_H
#define HELIOS_INITIALIZERADIATION_H

// Forward Declaration
class RadiationModel;
class SolarPosition;

#include "RadiationModel.h"
#include "SolarPosition.h"

void InitializeRadiation(const std::string &xml_input_file, SolarPosition *solarposition_ptr, RadiationModel *radiation_ptr, helios::Context *context_ptr);

#endif // HELIOS_INITIALIZERADIATION_H
