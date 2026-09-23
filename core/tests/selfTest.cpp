/**
 * \file "selfTest.cpp" Context selfTest() function.
 *
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

#include "Context.h"

#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_DISABLE_AUTOMATIC_DISCOVERY
#include "doctest.h"
#include "doctest_utils.h"

using namespace helios;

double errtol = 1e-6;

#include "Test_OBJ.h"
#include "Test_XML.h"
#include "Test_context.h"
#include "Test_data.h"
#include "Test_functions.h"
#include "Test_utilities.h"

int Context::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
