/** \file "TestMain.cpp" Test runner entry point for SoilTransportModel plugin.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilTransportModel.h"

using namespace helios;

int main(int argc, char **argv) {
    return SoilTransportModel::selfTest(argc, argv);
}
