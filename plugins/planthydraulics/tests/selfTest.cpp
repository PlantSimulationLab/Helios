/** \file "selfTest.cpp" Automated tests for plant hydraulics plug-in.

Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PlantHydraulicsModel.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace std;
using namespace helios;

float err_tol = 1e-5;

DOCTEST_TEST_CASE("PlantHydraulicsModel - Constructor") {
    Context context_test;
    DOCTEST_CHECK_NOTHROW(PlantHydraulicsModel ph_test_constructor(&context_test));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - setModelCoefficients") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    PlantHydraulicsModelCoefficients coeffs;
    coeffs.setLeafHydraulicCapacitance(-1.0, 0.5, 1.0);
    DOCTEST_CHECK_NOTHROW(ph_test.setModelCoefficients(coeffs));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - setModelCoefficientsFromLibrary") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    DOCTEST_CHECK_NOTHROW(ph_test.setModelCoefficientsFromLibrary("Walnut"));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - getModelCoefficientsFromLibrary") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    PlantHydraulicsModelCoefficients coeffs;
    DOCTEST_CHECK_NOTHROW(coeffs = ph_test.getModelCoefficientsFromLibrary("Walnut"));
    // Note: Cannot directly access private LeafHydraulicCapacitance member
    // Test passes if no exception is thrown during library access
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - getOrInitializePrimitiveData") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    uint test_uuid = context_test.addPatch();

    // Test getting existing data
    context_test.setPrimitiveData(test_uuid, "test_data", 123.45f);
    float retrieved_data = ph_test.getOrInitializePrimitiveData(test_uuid, "test_data", 0.0f, false);
    DOCTEST_CHECK(retrieved_data == doctest::Approx(123.45f).epsilon(err_tol));

    // Test initializing non-existing data
    float initialized_data = ph_test.getOrInitializePrimitiveData(test_uuid, "new_data", 99.9f, false);
    DOCTEST_CHECK(initialized_data == doctest::Approx(99.9f).epsilon(err_tol));
}

// Note: adjustTimestep is a private method and cannot be tested directly
// This functionality is tested indirectly through the run() method in the documentation examples

DOCTEST_TEST_CASE("PlantHydraulicsModel - outputConductancePrimitiveData") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    DOCTEST_CHECK_NOTHROW(ph_test.outputConductancePrimitiveData(true));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - outputCapacitancePrimitiveData") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    DOCTEST_CHECK_NOTHROW(ph_test.outputCapacitancePrimitiveData(true));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - setSoilWaterPotentialOfPlant & getSoilWaterPotentialOfPlant") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    uint test_plant_id = 100;
    float test_potential = -0.75f;

    DOCTEST_CHECK_NOTHROW(ph_test.setSoilWaterPotentialOfPlant(test_plant_id, test_potential));
    float retrieved_potential;
    DOCTEST_CHECK_NOTHROW(retrieved_potential = ph_test.getSoilWaterPotentialOfPlant(test_plant_id));
    DOCTEST_CHECK(retrieved_potential == doctest::Approx(test_potential).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - getPlantID") {
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);

    // Create a dummy primitive and associate it with a plant object
    uint primitive_uuid = context_test.addPatch();
    uint plant_object_id = context_test.addPolymeshObject({primitive_uuid});
    int expected_plant_id = 500;
    context_test.setObjectData(plant_object_id, "plantID", expected_plant_id);

    int retrieved_plant_id;
    DOCTEST_CHECK_NOTHROW(retrieved_plant_id = ph_test.getPlantID(primitive_uuid));
    DOCTEST_CHECK(retrieved_plant_id == expected_plant_id);
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - getUniquePlantIDs") {
    Context context_test;
    std::vector<uint> leaves;
    vec3 center = vec3(0, 0, 0);
    vec2 size = vec2(0.1, 0.1);
    uint leaf1 = context_test.addPatch(center, size);
    context_test.setPrimitiveData(leaf1, "plantID", 1);
    uint leaf2 = context_test.addPatch(center + vec3(0.1, 0.1, 0.1), size);
    context_test.setPrimitiveData(leaf2, "plantID", 2);
    leaves = {leaf1, leaf2};

    PlantHydraulicsModel hydraulics(&context_test);
    std::vector<int> IDs;
    DOCTEST_CHECK_NOTHROW(IDs = hydraulics.getUniquePlantIDs(leaves));
    int check11 = IDs.front();
    int check12 = IDs.back();
    int check21 = hydraulics.getPlantID(leaf1);
    int check22 = hydraulics.getPlantID(leaf2);

    DOCTEST_CHECK((check11 == check21 || check11 == check22));
    DOCTEST_CHECK((check12 == check21 || check12 == check22));
    DOCTEST_CHECK(check11 != check12);
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - Documentation Example 1") {
    Context context_test;

    uint objID = context_test.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(5, 5));
    std::vector<uint> leaves = context_test.getObjectPrimitiveUUIDs(objID);
    context_test.setObjectData(objID, "plantID", 1);
    context_test.setPrimitiveData(leaves, "latent_flux", 100.f);

    PlantHydraulicsModel hydraulics(&context_test);
    PlantHydraulicsModelCoefficients phmc;

    DOCTEST_CHECK_NOTHROW(phmc.setLeafHydraulicCapacitanceFromLibrary("pistachio"));
    DOCTEST_CHECK_NOTHROW(hydraulics.setModelCoefficients(phmc));

    int plantID = hydraulics.getPlantID(leaves);
    float soil_water_potential = -0.05;

    DOCTEST_CHECK_NOTHROW(hydraulics.setSoilWaterPotentialOfPlant(plantID, soil_water_potential));
    DOCTEST_CHECK_NOTHROW(hydraulics.run(leaves));

    for (uint UUID: leaves) {
        float stem_potential = hydraulics.getStemWaterPotential(UUID);
        DOCTEST_CHECK(stem_potential == doctest::Approx(-0.0590909).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - Documentation Example 2") {
    Context context_test;

    uint objID = context_test.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(5, 5));
    std::vector<uint> leaves = context_test.getObjectPrimitiveUUIDs(objID);
    context_test.setObjectData(objID, "plantID", 1);
    context_test.setPrimitiveData(leaves, "latent_flux", 100.f);

    PlantHydraulicsModel hydraulics(&context_test);
    PlantHydraulicsModelCoefficients phmc;

    DOCTEST_CHECK_NOTHROW(phmc.setLeafHydraulicCapacitanceFromLibrary("pistachio"));
    DOCTEST_CHECK_NOTHROW(hydraulics.setModelCoefficients(phmc));

    int plantID = hydraulics.getPlantID(leaves);
    float soil_water_potential = -0.05;

    DOCTEST_CHECK_NOTHROW(hydraulics.setSoilWaterPotentialOfPlant(plantID, soil_water_potential));
    DOCTEST_CHECK_NOTHROW(hydraulics.run(leaves));

    // Query the point-wise water potentials throughout the system
    for (int i = 0; i < leaves.size(); i++) {
        uint leafUUID = leaves.at(i);
        float psi_soil, psi_root, psi_stem;
        DOCTEST_CHECK_NOTHROW(psi_soil = hydraulics.getSoilWaterPotential(leafUUID));
        DOCTEST_CHECK_NOTHROW(psi_root = hydraulics.getRootWaterPotential(leafUUID));
        DOCTEST_CHECK_NOTHROW(psi_stem = hydraulics.getStemWaterPotential(leafUUID));

        // Check that values are finite and reasonable
        DOCTEST_CHECK(std::isfinite(psi_soil));
        DOCTEST_CHECK(std::isfinite(psi_root));
        DOCTEST_CHECK(std::isfinite(psi_stem));
    }
}

int PlantHydraulicsModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
