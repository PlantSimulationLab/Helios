/** \file "selfTest.cpp" Automated tests for plant hydraulics plug-in.

Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

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
    HydraulicCapacitance leaf_capacitance = coeffs.getLeafHydraulicCapacitance();
    DOCTEST_CHECK(leaf_capacitance.osmotic_potential_at_full_turgor == doctest::Approx(-1.6386f).epsilon(err_tol));
    DOCTEST_CHECK(leaf_capacitance.relative_water_content_at_turgor_loss == doctest::Approx(0.7683f).epsilon(err_tol));
    DOCTEST_CHECK(leaf_capacitance.cell_wall_elasticity_exponent == doctest::Approx(2.f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PlantHydraulicsModelCoefficients - getters return what the setters stored") {
    PlantHydraulicsModelCoefficients coeffs;
    coeffs.setLeafHydraulicConductance(0.05f, -1.5f, 2.f, true);
    coeffs.setStemHydraulicConductance(0.3f, -2.5f);
    coeffs.setRootHydraulicConductance(0.7f);
    coeffs.setLeafHydraulicCapacitance(-2.2f, 0.85f, 1.5f, 12.f);
    coeffs.setStemHydraulicCapacitance(0.4f);
    coeffs.setRootHydraulicCapacitance(-1.1f, 0.9f);

    HydraulicConductance K_leaf = coeffs.getLeafHydraulicConductance();
    DOCTEST_CHECK(K_leaf.saturated_conductance == doctest::Approx(0.05f).epsilon(err_tol));
    DOCTEST_CHECK(K_leaf.potential_at_half_saturated == doctest::Approx(-1.5f).epsilon(err_tol));
    DOCTEST_CHECK(K_leaf.sensitivity == doctest::Approx(2.f).epsilon(err_tol));
    DOCTEST_CHECK(K_leaf.temperature_dependence);

    HydraulicConductance K_stem = coeffs.getStemHydraulicConductance();
    DOCTEST_CHECK(K_stem.saturated_conductance == doctest::Approx(0.3f).epsilon(err_tol));
    DOCTEST_CHECK(K_stem.potential_at_half_saturated == doctest::Approx(-2.5f).epsilon(err_tol));
    DOCTEST_CHECK(!K_stem.temperature_dependence);

    HydraulicConductance K_root = coeffs.getRootHydraulicConductance();
    DOCTEST_CHECK(K_root.saturated_conductance == doctest::Approx(0.7f).epsilon(err_tol));

    HydraulicCapacitance C_leaf = coeffs.getLeafHydraulicCapacitance();
    DOCTEST_CHECK(C_leaf.osmotic_potential_at_full_turgor == doctest::Approx(-2.2f).epsilon(err_tol));
    DOCTEST_CHECK(C_leaf.relative_water_content_at_turgor_loss == doctest::Approx(0.85f).epsilon(err_tol));
    DOCTEST_CHECK(C_leaf.cell_wall_elasticity_exponent == doctest::Approx(1.5f).epsilon(err_tol));
    DOCTEST_CHECK(C_leaf.saturated_specific_water_content == doctest::Approx(12.f).epsilon(err_tol));

    HydraulicCapacitance C_stem = coeffs.getStemHydraulicCapacitance();
    DOCTEST_CHECK(C_stem.fixed_constant_capacitance == doctest::Approx(0.4f).epsilon(err_tol));

    HydraulicCapacitance C_root = coeffs.getRootHydraulicCapacitance();
    DOCTEST_CHECK(C_root.osmotic_potential_at_full_turgor == doctest::Approx(-1.1f).epsilon(err_tol));
    DOCTEST_CHECK(C_root.relative_water_content_at_turgor_loss == doctest::Approx(0.9f).epsilon(err_tol));
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

DOCTEST_TEST_CASE("PlantHydraulicsModel - getPlantID of an empty primitive list raises an error") {
    // Plant 0 is a valid plant, so an empty list must not be answered with a plant ID
    Context context_test;
    PlantHydraulicsModel ph_test(&context_test);
    uint leaf = context_test.addPatch();
    context_test.setPrimitiveData(leaf, "plantID", 0);
    ph_test.setSoilWaterPotentialOfPlant(0, -0.05f);

    const std::vector<uint> no_primitives;
    auto errorFrom = [](const auto &call) {
        try {
            call();
        } catch (const std::runtime_error &error) {
            return std::string(error.what());
        }
        return std::string("no error");
    };
    std::string plant_id_error, potential_error, printed;
    {
        helios::capture_cout cout_capture;
        plant_id_error = errorFrom([&] { static_cast<void>(ph_test.getPlantID(no_primitives)); });
        potential_error = errorFrom([&] { static_cast<void>(ph_test.getSoilWaterPotential(no_primitives)); });
        printed = cout_capture.get_captured_output();
    }
    DOCTEST_CHECK(plant_id_error.find("empty") != std::string::npos);
    DOCTEST_CHECK(potential_error.find("empty") != std::string::npos);
    DOCTEST_CHECK(printed.empty());
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

DOCTEST_TEST_CASE("PlantHydraulicsModel - getPrimitivesByPlantIDs matches getPrimitivesByPlantID") {
    // Several plants, identified by primitive data and by parent object data, plus primitives of no plant: the
    // one-pass batch lookup must return exactly what the per-plant lookup returns.
    Context context_test;
    PlantHydraulicsModel hydraulics(&context_test);
    for (int plant = 1; plant <= 3; plant++)
        for (int i = 0; i < 4; i++)
            context_test.setPrimitiveData(context_test.addPatch(vec3(plant, i, 0), vec2(0.1, 0.1)), "plantID", plant);
    std::vector<uint> obj_leaves;
    for (int i = 0; i < 3; i++)
        obj_leaves.push_back(context_test.addPatch(vec3(9, i, 0), vec2(0.1, 0.1)));
    context_test.setObjectData(context_test.addPolymeshObject(obj_leaves), "plantID", 7);
    context_test.addPatch(vec3(5, 5, 0), vec2(0.1, 0.1)); // no plant
    const std::vector<int> ids = {1, 2, 3, 7, 42}; // 42 has no primitives
    const auto byPlant = hydraulics.getPrimitivesByPlantIDs(ids);
    DOCTEST_CHECK(byPlant.size() == ids.size());
    for (int id: ids)
        DOCTEST_CHECK(byPlant.at(id) == hydraulics.getPrimitivesByPlantID(id));
    DOCTEST_CHECK(byPlant.at(7).size() == 3);
    DOCTEST_CHECK(byPlant.at(42).empty());
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

DOCTEST_TEST_CASE("PlantHydraulicsModel - stem and root water potentials are independent of plant leaf area") {
    // Two plants with identical per-area transpiration, coefficients and soil water potential but different total leaf
    // area (1 m^2 vs 4 m^2). The plant-level transpiration in the governing equations is the area-weighted MEAN flux
    // (W/m^2), and the conductances are per unit leaf area, so the steady-state stem and root potentials must agree.
    Context context_test;

    uint objID_small = context_test.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(5, 5));
    uint objID_large = context_test.addTileObject(make_vec3(10, 0, 0), make_vec2(2, 2), nullrotation, make_int2(5, 5));
    std::vector<uint> leaves_small = context_test.getObjectPrimitiveUUIDs(objID_small);
    std::vector<uint> leaves_large = context_test.getObjectPrimitiveUUIDs(objID_large);
    context_test.setObjectData(objID_small, "plantID", 1);
    context_test.setObjectData(objID_large, "plantID", 2);
    context_test.setPrimitiveData(leaves_small, "latent_flux", 100.f);
    context_test.setPrimitiveData(leaves_large, "latent_flux", 100.f);

    float area_small, area_large;
    DOCTEST_CHECK_NOTHROW(area_small = context_test.sumPrimitiveSurfaceArea(leaves_small));
    DOCTEST_CHECK_NOTHROW(area_large = context_test.sumPrimitiveSurfaceArea(leaves_large));
    DOCTEST_CHECK(area_small == doctest::Approx(1.f).epsilon(err_tol));
    DOCTEST_CHECK(area_large == doctest::Approx(4.f).epsilon(err_tol));

    PlantHydraulicsModel hydraulics(&context_test);
    PlantHydraulicsModelCoefficients phmc;
    DOCTEST_CHECK_NOTHROW(phmc.setLeafHydraulicCapacitanceFromLibrary("pistachio"));
    DOCTEST_CHECK_NOTHROW(hydraulics.setModelCoefficients(phmc));

    float soil_water_potential = -0.05;
    DOCTEST_CHECK_NOTHROW(hydraulics.setSoilWaterPotentialOfPlant(1, soil_water_potential));
    DOCTEST_CHECK_NOTHROW(hydraulics.setSoilWaterPotentialOfPlant(2, soil_water_potential));

    std::vector<uint> all_leaves = leaves_small;
    all_leaves.insert(all_leaves.end(), leaves_large.begin(), leaves_large.end());
    DOCTEST_CHECK_NOTHROW(hydraulics.run(all_leaves));

    float psi_stem_small = hydraulics.getStemWaterPotentialOfPlant(1);
    float psi_stem_large = hydraulics.getStemWaterPotentialOfPlant(2);
    float psi_root_small = hydraulics.getRootWaterPotentialOfPlant(1);
    float psi_root_large = hydraulics.getRootWaterPotentialOfPlant(2);

    // The 1 m^2 plant reproduces the documented steady-state value from Documentation Example 1
    DOCTEST_CHECK(psi_stem_small == doctest::Approx(-0.0590909).epsilon(err_tol));

    // The 4 m^2 plant must give the same answer: a whole-plant SUM of latent flux would make its potential drop 4x larger
    DOCTEST_CHECK(psi_stem_large == doctest::Approx(psi_stem_small).epsilon(err_tol));
    DOCTEST_CHECK(psi_root_large == doctest::Approx(psi_root_small).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - computeCapacitance first derivative correctness") {
    // Bug 1: computeCapacitance was using the second derivative stencil (f0 - 2f1 + f2)/(h^2)
    // instead of the first derivative stencil (f2 - f0)/(2h).
    // Capacitance C = Wsat / (dpsi/dw), which requires the first derivative.
    HydraulicCapacitance coeffs(-1.6386f, 0.7683f, 2.f); // Walnut parameters
    float w = 0.9f; // test at high relative water content
    float C = computeCapacitance(coeffs, w);

    // Manually compute expected capacitance using correct first derivative
    float h = 0.001f;
    float f0 = computeWaterPotential(coeffs, w - h);
    float f2 = computeWaterPotential(coeffs, w + h);
    float dpsidw_correct = (f2 - f0) / (2.f * h);
    float Wsat = coeffs.saturated_specific_water_content;
    float C_expected = Wsat / dpsidw_correct;

    DOCTEST_CHECK(C == doctest::Approx(C_expected).epsilon(1e-3));
    // The capacitance should be positive and finite for a valid PV curve
    DOCTEST_CHECK(std::isfinite(C));
    // With the bug (second derivative), the value is orders of magnitude different
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - setStemHydraulicConductanceTemperatureDependence modifies stem not root") {
    // Bug 3: setStemHydraulicConductanceTemperatureDependence was modifying
    // RootHydraulicConductance instead of StemHydraulicConductance.
    // Test strategy: Use constant (non-vulnerable) conductances so K = Ksat * temp_factor.
    // Enable stem temp dependence only. Run at two different temperatures.
    // Compare: with the fix, stem potential differs; with the bug, root potential differs instead.

    // We test directly with computeConductance since it's a free function.
    // With stem temp dependence ON, computeConductance(StemHC, psi, T1) != computeConductance(StemHC, psi, T2).
    // But with the bug, the stem struct doesn't have temperature_dependence set, so they'd be equal.
    HydraulicConductance stem_cond(0.5f, 0.f, 0.f); // constant Ksat=0.5, no vulnerability
    HydraulicConductance root_cond(0.5f, 0.f, 0.f);

    // Simulate what setStemHydraulicConductanceTemperatureDependence does
    // The buggy code: sets root_cond.temperature_dependence = true instead of stem_cond
    PlantHydraulicsModelCoefficients coeffs;
    coeffs.setStemHydraulicConductance(0.5f, 0.f, 0.f);
    coeffs.setRootHydraulicConductance(0.5f, 0.f, 0.f);
    coeffs.setStemHydraulicConductanceTemperatureDependence(true);

    // Now run two simulations with different temperatures and check that stem conductance differs
    Context ctx1;
    uint obj1 = ctx1.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
    std::vector<uint> leaves1 = ctx1.getObjectPrimitiveUUIDs(obj1);
    ctx1.setObjectData(obj1, "plantID", 1);
    ctx1.setPrimitiveData(leaves1, "latent_flux", 100.f);
    ctx1.setPrimitiveData(leaves1, "air_temperature", 298.15f);

    PlantHydraulicsModel hydro1(&ctx1);
    hydro1.setModelCoefficients(coeffs);
    hydro1.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydro1.outputConductancePrimitiveData(true);
    hydro1.run(leaves1);
    float stem_wp1 = hydro1.getStemWaterPotential(leaves1.front());
    float root_wp1 = hydro1.getRootWaterPotential(leaves1.front());

    Context ctx2;
    uint obj2 = ctx2.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
    std::vector<uint> leaves2 = ctx2.getObjectPrimitiveUUIDs(obj2);
    ctx2.setObjectData(obj2, "plantID", 1);
    ctx2.setPrimitiveData(leaves2, "latent_flux", 100.f);
    ctx2.setPrimitiveData(leaves2, "air_temperature", 320.f); // much hotter

    PlantHydraulicsModel hydro2(&ctx2);
    hydro2.setModelCoefficients(coeffs);
    hydro2.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydro2.outputConductancePrimitiveData(true);
    hydro2.run(leaves2);
    float stem_wp2 = hydro2.getStemWaterPotential(leaves2.front());
    float root_wp2 = hydro2.getRootWaterPotential(leaves2.front());

    // In steady-state: root_wp = soil_wp - E/K_root, stem_wp = root_wp - E/K_stem
    // With the bug, stem has NO temp dependence (K_stem is constant), so the difference
    // stem_wp - root_wp = -E/K_stem is the same at both temperatures.
    // With the fix, stem HAS temp dependence, so -E/K_stem differs.
    float stem_drop1 = stem_wp1 - root_wp1; // = -E/K_stem at T1
    float stem_drop2 = stem_wp2 - root_wp2; // = -E/K_stem at T2

    // With fix: stem_drop1 != stem_drop2 because K_stem is temperature-dependent
    DOCTEST_CHECK(stem_drop1 != doctest::Approx(stem_drop2).epsilon(1e-4));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - PistachioFemale library parameter consistency") {
    // Bug 4: setLeafHydraulicCapacitanceFromLibrary had wrong parameters for PistachioFemale:
    // osmotic potential was positive (physically impossible) and RWC at turgor loss was Walnut's value.
    // Test: both code paths should produce the same leaf water potential.
    Context context_header;
    uint objH = context_header.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
    std::vector<uint> leavesH = context_header.getObjectPrimitiveUUIDs(objH);
    context_header.setObjectData(objH, "plantID", 1);
    context_header.setPrimitiveData(leavesH, "latent_flux", 100.f);

    PlantHydraulicsModel hydro_header(&context_header);
    PlantHydraulicsModelCoefficients coeffs_header;
    coeffs_header.setLeafHydraulicCapacitanceFromLibrary("PistachioFemale");
    hydro_header.setModelCoefficients(coeffs_header);
    hydro_header.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydro_header.run(leavesH);
    float leaf_wp_header;
    context_header.getPrimitiveData(leavesH.front(), "water_potential", leaf_wp_header);

    Context context_cpp;
    uint objC = context_cpp.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
    std::vector<uint> leavesC = context_cpp.getObjectPrimitiveUUIDs(objC);
    context_cpp.setObjectData(objC, "plantID", 1);
    context_cpp.setPrimitiveData(leavesC, "latent_flux", 100.f);

    PlantHydraulicsModel hydro_cpp(&context_cpp);
    hydro_cpp.setModelCoefficientsFromLibrary("PistachioFemale");
    hydro_cpp.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydro_cpp.run(leavesC);
    float leaf_wp_cpp;
    context_cpp.getPrimitiveData(leavesC.front(), "water_potential", leaf_wp_cpp);

    // Both paths should produce the same leaf water potential
    DOCTEST_CHECK(leaf_wp_header == doctest::Approx(leaf_wp_cpp).epsilon(1e-4));

    // Leaf water potential should be negative (physically required)
    DOCTEST_CHECK(leaf_wp_header < 0.f);

    // The PV curve decomposition (turgor, osmotic, RWC) should also match between the two paths
    float rwc_header, rwc_cpp, turgor_header, turgor_cpp, osmotic_header, osmotic_cpp;
    context_header.getPrimitiveData(leavesH.front(), "relative_water_content", rwc_header);
    context_cpp.getPrimitiveData(leavesC.front(), "relative_water_content", rwc_cpp);
    context_header.getPrimitiveData(leavesH.front(), "turgor_pressure", turgor_header);
    context_cpp.getPrimitiveData(leavesC.front(), "turgor_pressure", turgor_cpp);
    context_header.getPrimitiveData(leavesH.front(), "osmotic_potential", osmotic_header);
    context_cpp.getPrimitiveData(leavesC.front(), "osmotic_potential", osmotic_cpp);

    DOCTEST_CHECK(rwc_header == doctest::Approx(rwc_cpp).epsilon(1e-3));
    DOCTEST_CHECK(turgor_header == doctest::Approx(turgor_cpp).epsilon(1e-3));
    DOCTEST_CHECK(osmotic_header == doctest::Approx(osmotic_cpp).epsilon(1e-3));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - default species uses Walnut values") {
    // Bug 4b: The default/unknown species case in setLeafHydraulicCapacitanceFromLibrary
    // used Western Redbud values but claimed to be Walnut.
    Context context_walnut;
    uint objW = context_walnut.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
    std::vector<uint> leavesW = context_walnut.getObjectPrimitiveUUIDs(objW);
    context_walnut.setObjectData(objW, "plantID", 1);
    context_walnut.setPrimitiveData(leavesW, "latent_flux", 100.f);

    PlantHydraulicsModel hydro_walnut(&context_walnut);
    PlantHydraulicsModelCoefficients coeffs_walnut;
    coeffs_walnut.setLeafHydraulicCapacitanceFromLibrary("Walnut");
    hydro_walnut.setModelCoefficients(coeffs_walnut);
    hydro_walnut.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydro_walnut.run(leavesW);
    float leaf_wp_walnut;
    context_walnut.getPrimitiveData(leavesW.front(), "water_potential", leaf_wp_walnut);

    Context context_default;
    uint objD = context_default.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
    std::vector<uint> leavesD = context_default.getObjectPrimitiveUUIDs(objD);
    context_default.setObjectData(objD, "plantID", 1);
    context_default.setPrimitiveData(leavesD, "latent_flux", 100.f);

    PlantHydraulicsModel hydro_default(&context_default);
    PlantHydraulicsModelCoefficients coeffs_default;
    {
        capture_cout capture;
        coeffs_default.setLeafHydraulicCapacitanceFromLibrary("UnknownSpecies");
    }
    hydro_default.setModelCoefficients(coeffs_default);
    hydro_default.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydro_default.run(leavesD);
    float leaf_wp_default;
    context_default.getPrimitiveData(leavesD.front(), "water_potential", leaf_wp_default);

    // Default should match Walnut leaf water potential
    DOCTEST_CHECK(leaf_wp_default == doctest::Approx(leaf_wp_walnut).epsilon(1e-4));

    // PV curve decomposition should also match
    float rwc_walnut, rwc_default, turgor_walnut, turgor_default;
    context_walnut.getPrimitiveData(leavesW.front(), "relative_water_content", rwc_walnut);
    context_default.getPrimitiveData(leavesD.front(), "relative_water_content", rwc_default);
    context_walnut.getPrimitiveData(leavesW.front(), "turgor_pressure", turgor_walnut);
    context_default.getPrimitiveData(leavesD.front(), "turgor_pressure", turgor_default);

    DOCTEST_CHECK(rwc_default == doctest::Approx(rwc_walnut).epsilon(1e-3));
    DOCTEST_CHECK(turgor_default == doctest::Approx(turgor_walnut).epsilon(1e-3));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - per-primitive model coefficients lookup") {
    // Bug 5: modelcoeffs_map lookup used loop index i instead of UUID.
    // Per-primitive coefficients set via setModelCoefficients(coeffs, UUIDs) would silently fail.
    // Strategy: Create many dummy primitives first to push UUIDs well above 0,1,2,...
    // so loop indices won't accidentally match UUIDs.
    Context context_test;

    // Add dummy primitives to push UUID counter high (ensures leaf UUIDs != loop indices)
    for (int k = 0; k < 100; k++) {
        context_test.addPatch();
    }

    uint objID = context_test.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));
    std::vector<uint> leaves = context_test.getObjectPrimitiveUUIDs(objID);
    context_test.setObjectData(objID, "plantID", 1);
    context_test.setPrimitiveData(leaves, "latent_flux", 100.f);

    PlantHydraulicsModel hydraulics(&context_test);

    // Set coefficients with different leaf conductances
    PlantHydraulicsModelCoefficients coeffs_high_k;
    coeffs_high_k.setLeafHydraulicCapacitanceFromLibrary("Walnut");
    coeffs_high_k.setLeafHydraulicConductance(1.0f); // high conductance

    PlantHydraulicsModelCoefficients coeffs_low_k;
    coeffs_low_k.setLeafHydraulicCapacitanceFromLibrary("Walnut");
    coeffs_low_k.setLeafHydraulicConductance(0.1f); // low conductance

    // Set different conductances for each group
    std::vector<uint> first_half(leaves.begin(), leaves.begin() + 2);
    std::vector<uint> second_half(leaves.begin() + 2, leaves.end());
    hydraulics.setModelCoefficients(coeffs_high_k, first_half);
    hydraulics.setModelCoefficients(coeffs_low_k, second_half);

    hydraulics.setSoilWaterPotentialOfPlant(1, -0.05f);
    hydraulics.run(leaves);

    // With different leaf conductances, leaf WP = stem_wp - E/K_leaf should differ.
    float wp_first, wp_second;
    context_test.getPrimitiveData(first_half.at(0), "water_potential", wp_first);
    context_test.getPrimitiveData(second_half.at(0), "water_potential", wp_second);

    // With bug (loop index lookup), modelcoeffs_map.find(0) != find(UUID>100), so both use defaults.
    // With fix (UUID lookup), each group uses its own coefficients.
    DOCTEST_CHECK(wp_first != doctest::Approx(wp_second).epsilon(1e-4));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - computeConductance fabs correctness") {
    // Bug 7: abs() on float may resolve to integer overload, truncating the value.
    // Test with negative water potential and non-zero vulnerability parameters.
    HydraulicConductance coeffs(0.5f, -2.0f, 3.0f); // Ksat=0.5, a=-2.0, b=3.0
    float psi = -1.5f;
    float T = 298.15f;
    float K = computeConductance(coeffs, psi, T);

    // Expected: K = Ksat / (1 + |psi/a|^b) = 0.5 / (1 + |(-1.5)/(-2.0)|^3) = 0.5 / (1 + 0.75^3) = 0.5 / 1.421875
    float expected = 0.5f / (1.f + powf(fabsf(-1.5f / -2.0f), 3.0f));
    DOCTEST_CHECK(K == doctest::Approx(expected).epsilon(1e-5));
}

DOCTEST_TEST_CASE("PlantHydraulicsModel - non-steady-state leaf dynamics normalization") {
    // Bug 2: Leaf transient update added mol/m² to dimensionless RWC without dividing by Wsat.
    // Bug 6: Adaptive timestep delta_psi could be negative (missing fabs).
    // Strategy: Run two simulations differing ONLY in Wsat (1 vs 10).
    // With correct normalization (dividing by Wsat), a larger water reservoir (Wsat=10)
    // should produce less negative leaf WP (more buffered). With the bug (no division),
    // Wsat doesn't affect the transient dynamics, so both produce the same leaf WP.

    auto run_transient = [](float Wsat) -> float {
        Context ctx;
        uint obj = ctx.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(3, 3));
        std::vector<uint> leaves = ctx.getObjectPrimitiveUUIDs(obj);
        ctx.setObjectData(obj, "plantID", 1);
        ctx.setPrimitiveData(leaves, "latent_flux", 200.f);
        ctx.setPrimitiveData(leaves, "relative_water_content", 0.95f);
        ctx.setPrimitiveData(leaves, "water_potential", -0.5f);

        PlantHydraulicsModel hydraulics(&ctx);
        PlantHydraulicsModelCoefficients coeffs;
        coeffs.setLeafHydraulicCapacitance(-1.6386f, 0.7683f, 2.f, Wsat);
        coeffs.setStemHydraulicCapacitance(0.1f);
        coeffs.setRootHydraulicCapacitance(0.1f);
        hydraulics.setModelCoefficients(coeffs);
        hydraulics.setSoilWaterPotentialOfPlant(1, -0.05f);

        hydraulics.run(leaves, 100, 10);

        float leaf_wp;
        ctx.getPrimitiveData(leaves.front(), "water_potential", leaf_wp);
        return leaf_wp;
    };

    float wp_wsat1 = run_transient(1.f);
    float wp_wsat10 = run_transient(10.f);

    // With the fix, Wsat=10 buffers the leaf water content (slower RWC change),
    // producing a less negative leaf WP than Wsat=1. With the bug, both are identical
    // because Wsat is not used in the update formula.
    DOCTEST_CHECK(wp_wsat1 != doctest::Approx(wp_wsat10).epsilon(1e-3));
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

DOCTEST_TEST_CASE("PlantHydraulicsModel - whole-plant hydraulic conductance K_plant and soil water potential outputs") {
    // K_plant is the root, stem, and leaf conductances in series (per unit leaf area); it is written on request, or when the Buckley et al. (2026) stomatal model asks for it
    const float K_root = 0.08f, K_stem = 0.05f, K_leaf = 0.02f; // mol/m^2/s/MPa, independent of water potential and temperature
    const float K_plant_expected = 1.f / (1.f / K_root + 1.f / K_stem + 1.f / K_leaf);
    const float psi_soil = -0.3f;

    auto setup = [&](Context &context, PlantHydraulicsModel &hydraulics) {
        uint objID = context.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));
        std::vector<uint> leaves = context.getObjectPrimitiveUUIDs(objID);
        context.setObjectData(objID, "plantID", 1);
        context.setPrimitiveData(leaves, "latent_flux", 100.f);
        PlantHydraulicsModelCoefficients coeffs;
        coeffs.setRootHydraulicConductance(K_root);
        coeffs.setStemHydraulicConductance(K_stem);
        coeffs.setLeafHydraulicConductance(K_leaf);
        coeffs.setRootHydraulicCapacitance(1.f); // fixed capacitances, needed by the time-stepped solution
        coeffs.setStemHydraulicCapacitance(1.f);
        hydraulics.setModelCoefficients(coeffs);
        hydraulics.setSoilWaterPotentialOfPlant(1, psi_soil);
        return leaves;
    };

    SUBCASE("not written by default") {
        Context context;
        PlantHydraulicsModel hydraulics(&context);
        std::vector<uint> leaves = setup(context, hydraulics);
        hydraulics.run(leaves);
        DOCTEST_CHECK(!context.doesPrimitiveDataExist(leaves.front(), "plant_hydraulic_conductance"));
        DOCTEST_CHECK(!context.doesPrimitiveDataExist(leaves.front(), "soil_water_potential"));
    }

    SUBCASE("requested with outputPlantConductancePrimitiveData, steady state and time-stepped") {
        for (int timespan: {0, 1000}) {
            Context context;
            PlantHydraulicsModel hydraulics(&context);
            std::vector<uint> leaves = setup(context, hydraulics);
            hydraulics.outputPlantConductancePrimitiveData(true);
            hydraulics.run(leaves, timespan, 100);
            for (uint UUID: leaves) {
                float K_plant, psi;
                DOCTEST_REQUIRE(context.doesPrimitiveDataExist(UUID, "plant_hydraulic_conductance"));
                context.getPrimitiveData(UUID, "plant_hydraulic_conductance", K_plant);
                context.getPrimitiveData(UUID, "soil_water_potential", psi);
                DOCTEST_CHECK(K_plant == doctest::Approx(K_plant_expected).epsilon(1e-5));
                DOCTEST_CHECK(psi == doctest::Approx(psi_soil).epsilon(1e-6));
            }
        }
    }

    SUBCASE("a soil water potential never set with setSoilWaterPotentialOfPlant() does not replace one set directly") {
        Context context;
        PlantHydraulicsModel hydraulics(&context);
        uint objID = context.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));
        std::vector<uint> leaves = context.getObjectPrimitiveUUIDs(objID);
        context.setObjectData(objID, "plantID", 2);
        context.setPrimitiveData(leaves, "latent_flux", 100.f);
        context.setPrimitiveData(leaves, "soil_water_potential", -1.5f); // set directly, e.g. from a soil model
        hydraulics.outputPlantConductancePrimitiveData(true);
        hydraulics.run(leaves);
        float psi;
        context.getPrimitiveData(leaves.front(), "soil_water_potential", psi);
        DOCTEST_CHECK(psi == doctest::Approx(-1.5f).epsilon(1e-6));
        DOCTEST_CHECK(context.doesPrimitiveDataExist(leaves.front(), "plant_hydraulic_conductance")); // K_plant is still written
    }

    SUBCASE("requested by the Buckley et al. (2026) stomatal model through global data") {
        Context context;
        PlantHydraulicsModel hydraulics(&context);
        std::vector<uint> leaves = setup(context, hydraulics);
        context.setGlobalData("buckley2026_stomatal_model_enabled", 1);
        hydraulics.run(leaves);
        float K_plant;
        DOCTEST_REQUIRE(context.doesPrimitiveDataExist(leaves.front(), "plant_hydraulic_conductance"));
        context.getPrimitiveData(leaves.front(), "plant_hydraulic_conductance", K_plant);
        DOCTEST_CHECK(K_plant == doctest::Approx(K_plant_expected).epsilon(1e-5));
        DOCTEST_CHECK(K_plant < K_leaf); // K_plant is not the leaf conductance
    }
}

int PlantHydraulicsModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
