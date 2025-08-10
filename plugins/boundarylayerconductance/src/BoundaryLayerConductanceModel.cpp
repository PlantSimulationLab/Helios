/** \file "BoundaryLayerConductanceModel.cpp" Boundary-layer conductance  model plugin declarations.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "BoundaryLayerConductanceModel.h"

using namespace helios;

BLConductanceModel::BLConductanceModel(helios::Context *__context) {

    wind_speed_default = 1.f;

    air_temperature_default = 293.f;

    surface_temperature_default = 303.f;

    message_flag = true; // print messages to screen by default

    // Copy pointer to the context
    context = __context;
}

void BLConductanceModel::enableMessages() {
    message_flag = true;
}

void BLConductanceModel::disableMessages() {
    message_flag = false;
}

void BLConductanceModel::setBoundaryLayerModel(const char *gH_model) {
    std::vector<uint> UUIDs = context->getAllUUIDs();
    setBoundaryLayerModel(UUIDs, gH_model);
}

void BLConductanceModel::setBoundaryLayerModel(uint UUID, const char *gH_model) {
    std::vector<uint> UUIDs{UUID};
    setBoundaryLayerModel(UUIDs, gH_model);
}

void BLConductanceModel::setBoundaryLayerModel(const std::vector<uint> &UUIDs, const char *gH_model) {

    uint model = 0;

    if (strcmp(gH_model, "Pohlhausen") == 0 || strcmp(gH_model, "Polhausen") == 0) {
        model = 0;
    } else if (strcmp(gH_model, "InclinedPlate") == 0) {
        model = 1;
    } else if (strcmp(gH_model, "Sphere") == 0) {
        model = 2;
    } else if (strcmp(gH_model, "Ground") == 0) {
        model = 3;
    } else {
        std::cerr << "WARNING (EnergyBalanceModel::setBoundaryLayerModel): Boundary-layer conductance model " << gH_model << " is unknown. Skipping this function call.." << std::endl;
        return;
    }

    for (uint UUID: UUIDs) {
        boundarylayer_model[UUID] = model;
    }
}

void BLConductanceModel::run() {

    run(context->getAllUUIDs());
}

void BLConductanceModel::run(const std::vector<uint> &UUIDs) {

    for (uint UUID: UUIDs) {

        float U;
        if (context->doesPrimitiveDataExist(UUID, "wind_speed")) {
            context->getPrimitiveData(UUID, "wind_speed", U);
        } else {
            U = wind_speed_default;
        }

        float Ta;
        if (context->doesPrimitiveDataExist(UUID, "air_temperature")) {
            context->getPrimitiveData(UUID, "air_temperature", Ta);
        } else {
            Ta = air_temperature_default;
        }

        float T;
        if (context->doesPrimitiveDataExist(UUID, "temperature")) {
            context->getPrimitiveData(UUID, "temperature", T);
        } else {
            T = surface_temperature_default;
        }

        float L;
        if (context->doesPrimitiveDataExist(UUID, "object_length")) {
            context->getPrimitiveData(UUID, "object_length", L);
            if (L == 0) {
                L = sqrt(context->getPrimitiveArea(UUID));
            }
        } else if (context->getPrimitiveParentObjectID(UUID) > 0) {
            uint objID = context->getPrimitiveParentObjectID(UUID);
            L = sqrt(context->getObjectArea(objID));
        } else {
            L = sqrt(context->getPrimitiveArea(UUID));
        }

        // Number of primitive faces
        char Nsides = 2; // default is 2
        if (context->doesPrimitiveDataExist(UUID, "twosided_flag") && context->getPrimitiveDataType("twosided_flag") == HELIOS_TYPE_UINT) {
            uint flag;
            context->getPrimitiveData(UUID, "twosided_flag", flag);
            if (flag == 0) {
                Nsides = 1;
            }
        }

        vec3 norm = context->getPrimitiveNormal(UUID);
        float inclination = cart2sphere(norm).zenith;

        float gH = calculateBoundaryLayerConductance(boundarylayer_model[UUID], U, L, Nsides, inclination, T, Ta);

        context->setPrimitiveData(UUID, "boundarylayer_conductance", gH);
    }
}

float BLConductanceModel::calculateBoundaryLayerConductance(uint gH_model, float U, float L, char Nsides, float inclination, float TL, float Ta) {


    assert(gH_model < 4);

    float gH = 0;

    if (L == 0) {
        return 0;
    }

    if (gH_model == 0) { // Pohlhausen equation
        // This comes from the correlation by Pohlhausen - see Eq. XX of Campbell and Norman (1998). It assumes a flat plate parallel to the direction of the flow, which extends infinitely in the cross-stream direction and "L" in the streamwise
        // direction. It also assumes that the air is at standard temperature and pressure, and flow is laminar, forced convection.

        gH = 0.135f * sqrt(U / L) * float(Nsides);

    } else if (gH_model == 1) { // Inclined Plate

        float Pr = 0.7f; // air Prandtl number

        float nu = 1.568e-5; // air viscosity (m^2/sec)
        float alpha = nu / Pr; // air diffusivity (m^2/sec)

        float Re = U * L / nu;

        float Gr = 9.81f * fabs(TL - Ta) * std::pow(L, 3) / ((Ta) *nu * nu);

        // For zero or very low wind speed, revert to pure free convection
        if (U <= 1e-6f || Re <= 1e-6f) {
            // Pure natural convection correlation for inclined plates
            float Nu_free;
            if (inclination < 75.f * M_PI / 180.f || inclination > 105.f * M_PI / 180.f) {
                // Nearly horizontal plates - use horizontal plate correlation
                Nu_free = 0.54f * std::pow(Gr * fabs(cos(inclination)), 0.25f);
            } else {
                // Nearly vertical plates - use vertical plate correlation
                Nu_free = 0.59f * std::pow(Gr, 0.25f);
            }
            gH = Nu_free * (alpha / L);
            return gH;
        }

        float F1 = 0.399f * std::pow(Pr, 1.f / 3.f) * pow(1.f + pow(0.0468f / Pr, 2.f / 3.f), -0.25f);
        float F2 = 0.75f * std::pow(Pr, 0.5f) * pow(2.5f * (1.f + 2.f * pow(Pr, 0.5f) + 2.f * Pr), -0.25f);

        // direction of free convection
        float free_direction;
        if (TL >= Ta) {
            free_direction = 1;
        } else {
            free_direction = -1;
        }

        // free_direction=1;

        if (inclination < 75.f * M_PI / 180.f || inclination > 105.f * M_PI / 180.f) {

            gH = 41.4f * alpha / L * 2.f * F1 * sqrtf(Re) * std::pow(1.f + free_direction * std::pow(2.f * F2 * std::pow(Gr * fabs(cos(inclination)) / (Re * Re), 0.25f) / (3.f * F1), 3), 1.f / 3.f);

        } else {

            float C = 0.07f * sqrtf(fabs(cos(inclination)));
            float F3 = std::pow(Pr, 0.5f) * std::pow(0.25f + 1.6f * std::pow(Pr, 0.5f), -1.f) * std::pow(Pr / 5.f, 1.f / 5.f + C);

            gH = 41.4f * alpha / L * 2.f * F1 * sqrtf(Re) * std::pow(1.f + free_direction * std::pow((F3 * pow(Gr * pow(Re, -5.f / 2.f), 1.f / 5.f) * std::pow(Gr, C)) / (6.f * (1.f / 5.f + C) * F1), 3.f), 1.f / 3.f);
        }

    } else if (gH_model == 2) { // Sphere

        // Laminar flow around a sphere (L is sphere diameter). From Eq. 4 of Smart and Sinclair (1976).

        float Pr = 0.7f; // air Prandtl number
        float nu = 1.568e-5; // air viscosity (m^2/sec)
        float ka = 0.024; // air thermal conductivity (W/m-K)
        float cp = 29.25; // air heat capacity (J/mol-K)

        float Re = U * L / nu;

        gH = (ka / cp) / L * (2.0 + 0.6 * sqrtf(Re) * pow(Pr, 1.f / 3.f));

    } else if (gH_model == 3) { // Ground
        // From Eq. A17 of Kustas and Norman (1999). For the soil-air interface.

        gH = 0.004f + 0.012f * U; // units in (m^3 air)/(m^2-sec.)

        // assuming standard temperature and pressure, (m^3 air)*(1.2041 kg/m^3)/(0.02897 kg/mol) = 41.56 (mol air)/(m^2-sec.)

        gH = gH * 41.56f;
    }

    return gH;
}
