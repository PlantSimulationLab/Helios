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

using namespace helios;

int PlantHydraulicsModel::selfTest() {
    std::cout << "Running Plant Hydraulics Model self-test..." << std::endl;

    int error_count = 0;
    double errtol = 1e-5;

    // Test 1: PlantHydraulicsModel Constructor
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test_constructor(&context);
            // No direct way to check constructor success other than it not throwing an exception
            // and subsequent methods working. We assume it passes if no exception is thrown.
            std::cout << "Test 1 (PlantHydraulicsModel Constructor): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 1 (PlantHydraulicsModel Constructor): Failed - " << e.what() << std::endl;
        }
    }

    // Test 2: setModelCoefficients (const PlantHydraulicsModelCoefficients &modelcoefficients)
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            PlantHydraulicsModelCoefficients coeffs;
            coeffs.setLeafHydraulicCapacitance(-1.0, 0.5, 1.0);
            ph_test.setModelCoefficients(coeffs);
            // No direct getter for modelcoeffs, so we assume it works if no exception is thrown.
            std::cout << "Test 2 (setModelCoefficients): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 2 (setModelCoefficients): Failed - " << e.what() << std::endl;
        }
    }

    // Test 3: setModelCoefficientsFromLibrary (const std::string &species)
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            ph_test.setModelCoefficientsFromLibrary("Walnut");
            // No direct getter for modelcoeffs, so we assume it works if no exception is thrown.
            std::cout << "Test 3 (setModelCoefficientsFromLibrary string): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 3 (setModelCoefficientsFromLibrary string): Failed - " << e.what() << std::endl;
        }
    }

    // Test 4: getModelCoefficientsFromLibrary (const std::string &species)
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            PlantHydraulicsModelCoefficients coeffs = ph_test.getModelCoefficientsFromLibrary("Walnut");
            if (std::abs(coeffs.LeafHydraulicCapacitance.osmotic_potential_at_full_turgor - (-1.6386)) > errtol) {
                error_count++;
                std::cerr << "Test 4 (getModelCoefficientsFromLibrary string): Failed - Incorrect coefficient value." << std::endl;
            } else {
                std::cout << "Test 4 (getModelCoefficientsFromLibrary string): Passed." << std::endl;
            }
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 4 (getModelCoefficientsFromLibrary string): Failed - " << e.what() << std::endl;
        }
    }

    // Test 5: getOrInitializePrimitiveData
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            uint test_uuid = context.addPatch(); // Create a dummy primitive

            // Test getting existing data
            context.setPrimitiveData(test_uuid, "test_data", 123.45f);
            float retrieved_data = ph_test.getOrInitializePrimitiveData(test_uuid, "test_data", 0.0f, false);
            if (std::abs(retrieved_data - 123.45f) > errtol) {
                error_count++;
                std::cerr << "Test 5 (getOrInitializePrimitiveData existing): Failed - Expected 123.45, got " << retrieved_data << std::endl;
            }

            // Test initializing non-existing data
            float initialized_data = ph_test.getOrInitializePrimitiveData(test_uuid, "new_data", 99.9f, false);
            if (std::abs(initialized_data - 99.9f) > errtol) {
                error_count++;
                std::cerr << "Test 5 (getOrInitializePrimitiveData new): Failed - Expected 99.9, got " << initialized_data << std::endl;
            }
            std::cout << "Test 5 (getOrInitializePrimitiveData): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 5 (getOrInitializePrimitiveData): Failed - " << e.what() << std::endl;
        }
    }

    // Test 6: adjustTimestep
    {
        try {
            PlantHydraulicsModel ph_test(nullptr); // Context not needed for this static-like function
            float time_step = 100.0f;
            float min_time_step = 10.0f;
            float max_time_step = 1000.0f;
            float gradient_upper_bound = 0.01f;

            // Test reduce timestep
            int adjusted_ts1 = ph_test.adjustTimestep(time_step, min_time_step, max_time_step, 0.02f, gradient_upper_bound);
            if (adjusted_ts1 >= time_step) {
                error_count++;
                std::cerr << "Test 6 (adjustTimestep reduce): Failed - Timestep not reduced." << std::endl;
            }

            // Test increase timestep
            int adjusted_ts2 = ph_test.adjustTimestep(time_step, min_time_step, max_time_step, 0.0005f, gradient_upper_bound);
            if (adjusted_ts2 <= time_step) {
                error_count++;
                std::cerr << "Test 6 (adjustTimestep increase): Failed - Timestep not increased." << std::endl;
            }

            // Test clamp to min
            int adjusted_ts3 = ph_test.adjustTimestep(5.0f, min_time_step, max_time_step, 0.02f, gradient_upper_bound);
            if (adjusted_ts3 != min_time_step) {
                error_count++;
                std::cerr << "Test 6 (adjustTimestep clamp min): Failed - Timestep not clamped to min." << std::endl;
            }

            // Test clamp to max
            int adjusted_ts4 = ph_test.adjustTimestep(2000.0f, min_time_step, max_time_step, 0.0005f, gradient_upper_bound);
            if (adjusted_ts4 != max_time_step) {
                error_count++;
                std::cerr << "Test 6 (adjustTimestep clamp max): Failed - Timestep not clamped to max." << std::endl;
            }
            std::cout << "Test 6 (adjustTimestep): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 6 (adjustTimestep): Failed - " << e.what() << std::endl;
        }
    }

    // Test 7: outputConductancePrimitiveData
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            ph_test.outputConductancePrimitiveData(true);
            // No direct getter to verify, assuming it works if no exception.
            std::cout << "Test 7 (outputConductancePrimitiveData): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 7 (outputConductancePrimitiveData): Failed - " << e.what() << std::endl;
        }
    }

    // Test 8: outputCapacitancePrimitiveData
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            ph_test.outputCapacitancePrimitiveData(true);
            // No direct getter to verify, assuming it works if no exception.
            std::cout << "Test 8 (outputCapacitancePrimitiveData): Passed." << std::endl;
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 8 (outputCapacitancePrimitiveData): Failed - " << e.what() << std::endl;
        }
    }

    // Test 9: setSoilWaterPotentialOfPlant & getSoilWaterPotentialOfPlant
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);
            uint test_plant_id = 100;
            float test_potential = -0.75f;

            ph_test.setSoilWaterPotentialOfPlant(test_plant_id, test_potential);
            float retrieved_potential = ph_test.getSoilWaterPotentialOfPlant(test_plant_id);

            if (std::abs(retrieved_potential - test_potential) > errtol) {
                error_count++;
                std::cerr << "Test 9 (set/getSoilWaterPotentialOfPlant): Failed - Expected " << test_potential << ", got " << retrieved_potential << std::endl;
            } else {
                std::cout << "Test 9 (set/getSoilWaterPotentialOfPlant): Passed." << std::endl;
            }
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 9 (set/getSoilWaterPotentialOfPlant): Failed - " << e.what() << std::endl;
        }
    }

    // Test 10: getPlantID (uint UUID)
    {
        try {
            Context context;
            PlantHydraulicsModel ph_test(&context);

            // Create a dummy primitive and associate it with a plant object
            uint primitive_uuid = context.addPatch();
            uint plant_object_id = context.addPolymeshObject({primitive_uuid});
            int expected_plant_id = 500; // Arbitrary ID
            context.setObjectData(plant_object_id, "plantID", expected_plant_id);

            int retrieved_plant_id = ph_test.getPlantID(primitive_uuid);

            if (retrieved_plant_id != expected_plant_id) {
                error_count++;
                std::cerr << "Test 10 (getPlantID uint): Failed - Expected " << expected_plant_id << ", got " << retrieved_plant_id << std::endl;
            } else {
                std::cout << "Test 10 (getPlantID uint): Passed." << std::endl;
            }
        } catch (const std::exception& e) {
            error_count++;
            std::cerr << "Test 10 (getPlantID uint): Failed - " << e.what() << std::endl;
        }
    }

    //Test 11: Run first code example in documentation
    {
        try {
            Context context;

            uint objID = context.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(5, 5));
            std::vector<uint> leaves = context.getObjectPrimitiveUUIDs(objID);
            context.setObjectData(objID, "plantID", 1);
            context.setPrimitiveData(leaves, "latent_flux", 100.f );

            PlantHydraulicsModel hydraulics(&context);
            PlantHydraulicsModelCoefficients phmc;

            phmc.setLeafHydraulicCapacitanceFromLibrary("pistachio");
            hydraulics.setModelCoefficients(phmc);

            int plantID = hydraulics.getPlantID(leaves);
            float soil_water_potential = -0.05; // MPa

            hydraulics.setSoilWaterPotentialOfPlant(plantID, soil_water_potential);
            hydraulics.run(leaves);

            std::vector<float> stem_water_potentials;
            bool failed = false;
            for ( uint UUID : leaves ) {
                if( fabs( hydraulics.getStemWaterPotential(UUID) + 0.0590909 ) > errtol ) {
                    failed = true;
                    break;
                }
            }
            if ( failed ) {
                std::cerr << "Test 11 (run doc example 1): Failed." << std::endl;
                error_count++;
            }else {
                std::cout << "Test 11 (run doc example 1): Passed." << std::endl;
            }
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 11 (run doc example 1): Failed - " << e.what() << std::endl;
        }
    }

    //Test 12: Run second code example in documentation
    {
        try {
            Context context;

            uint objID = context.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(5, 5));
            std::vector<uint> leaves = context.getObjectPrimitiveUUIDs(objID);
            context.setObjectData(objID, "plantID", 1);
            context.setPrimitiveData(leaves, "latent_flux", 100.f );

            PlantHydraulicsModel hydraulics(&context);
            PlantHydraulicsModelCoefficients phmc;

            phmc.setLeafHydraulicCapacitanceFromLibrary("pistachio");
            hydraulics.setModelCoefficients(phmc);

            int plantID = hydraulics.getPlantID(leaves);
            float soil_water_potential = -0.05; //MPa

            hydraulics.setSoilWaterPotentialOfPlant(plantID,soil_water_potential);
            hydraulics.run(leaves);

            // Query the point-wise water potentials throughout the system
            for( int i=0; i<leaves.size(); i++ ) {
                uint leafUUID = leaves.at(i);
                float psi_soil = hydraulics.getSoilWaterPotential(leafUUID);
                float psi_root = hydraulics.getRootWaterPotential(leafUUID);
                float psi_stem = hydraulics.getStemWaterPotential(leafUUID);
            }
            std::cout << "Test 12 (run doc example 2): Passed." << std::endl;
        } catch (const std::exception &e) {
            error_count++;
            std::cerr << "Test 12 (run doc example 2): Failed - " << e.what() << std::endl;
        }
    }

    if (error_count == 0) {
        std::cout << "All PlantHydraulicsModel self-tests passed!" << std::endl;
    } else {
        std::cerr << "PlantHydraulicsModel self-tests failed with " << error_count << " errors." << std::endl;
    }

    return error_count;
}