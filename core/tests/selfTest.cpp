/**
 * \file "selfTest.cpp" Context selfTest() function.
 *
 * Copyright (C) 2016-2025 Brian Bailey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "Context.h"

#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_DISABLE_AUTOMATIC_DISCOVERY
#include "doctest.h"

using namespace helios;

double errtol = 1e-6;

#include "Test_XML.h"
#include "Test_context.h"
#include "Test_data.h"
#include "Test_functions.h"
#include "Test_utilities.h"

int Context::selfTest() {

    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) { // important - query flags (and --exit) rely on this
        return res; // propagate the result of the tests
    }

    return res;
}
