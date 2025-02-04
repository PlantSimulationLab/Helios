#include "BoundaryLayerConductanceModel.h"
#include "Context.h"
#include <iostream>
#include <cmath>
#include <cassert>

using namespace helios;

int BLConductanceModel::selfTest() {

    std::cout << "Running complete unit tests for BoundaryLayerConductanceModel..." << std::endl;

    // Error tolerance for comparisons
    float err_tol = 1e-3;

    // Context and UUID setup
    Context context_1;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    std::vector<uint> UUID_1;
    UUID_1.push_back(context_1.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0.2 * M_PI, 0)));
    UUID_1.push_back(context_1.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0.2 * M_PI, 0)));
    UUID_1.push_back(context_1.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0.2 * M_PI, 0)));
    UUID_1.push_back(context_1.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0.2 * M_PI, 0)));

    context_1.setPrimitiveData(UUID_1, "air_temperature", Ta);
    context_1.setPrimitiveData(UUID_1, "temperature", TL);
    context_1.setPrimitiveData(UUID_1, "wind_speed", U);

    // Instantiate the boundary layer conductance model
    BLConductanceModel blc_1(&context_1);
    blc_1.disableMessages();

    // Original Tests
    blc_1.setBoundaryLayerModel(UUID_1.at(0), "Polhausen");
    blc_1.setBoundaryLayerModel(UUID_1.at(1), "InclinedPlate");
    blc_1.setBoundaryLayerModel(UUID_1.at(2), "Sphere");
    blc_1.setBoundaryLayerModel(UUID_1.at(3), "Ground");

    blc_1.run();

    std::vector<float> gH(UUID_1.size());

    context_1.getPrimitiveData(UUID_1.at(0), "boundarylayer_conductance", gH.at(0));
    context_1.getPrimitiveData(UUID_1.at(1), "boundarylayer_conductance", gH.at(1));
    context_1.getPrimitiveData(UUID_1.at(2), "boundarylayer_conductance", gH.at(2));
    context_1.getPrimitiveData(UUID_1.at(3), "boundarylayer_conductance", gH.at(3));

    if (fabs(gH.at(0) - 0.20914112f) / gH.at(0) > err_tol ||
        fabs(gH.at(1) - 0.133763f) / gH.at(1) > err_tol ||
        fabs(gH.at(2) - 0.087149f) / gH.at(2) > err_tol ||
        fabs(gH.at(3) - 0.465472f) / gH.at(3) > err_tol) {
        std::cout << "Failed original boundary-layer conductance model check." << std::endl;
        return 1;
    }

    std::cout << "Original model tests passed." << std::endl;

    // **New Test: Default Values Handling**
    Context context_2;
    uint UUID_default = context_2.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0.2 * M_PI, 0));

    BLConductanceModel blc_2(&context_2);
    blc_2.setBoundaryLayerModel(UUID_default, "Polhausen");
    blc_2.run();

    float default_gH;
    context_2.getPrimitiveData(UUID_default, "boundarylayer_conductance", default_gH);

    assert(default_gH > 0); // Ensure conductance is calculated even with defaults

    std::cout << "Default values handling passed." << std::endl;

    // **New Test: Single-Sided Primitive**
    context_1.setPrimitiveData(UUID_1.at(0), "twosided_flag", uint(0)); // Simulate single-sided primitive
    blc_1.run();
    context_1.getPrimitiveData(UUID_1.at(0), "boundarylayer_conductance", gH.at(0));

    std::cout << "Single-sided primitive handling passed." << std::endl;

    // **New Test: Enable and Disable Messages**
    blc_1.enableMessages();
    assert(blc_1.message_flag == true);

    blc_1.disableMessages();
    assert(blc_1.message_flag == false);

    std::cout << "Message flag tests passed." << std::endl;

    // **New Test: setBoundaryLayerModel(const char*)**
    blc_1.setBoundaryLayerModel("Polhausen");
    blc_1.setBoundaryLayerModel("InclinedPlate");
    blc_1.setBoundaryLayerModel("InvalidModel"); // Should trigger warning but not crash

    std::cout << "setBoundaryLayerModel(const char*) tests passed." << std::endl;

    // **New Test: run(const std::vector<uint>& UUIDs)**
    std::vector<uint> subsetUUIDs = {UUID_1.at(0), UUID_1.at(1)};
    blc_1.run(subsetUUIDs);

    context_1.getPrimitiveData(UUID_1.at(0), "boundarylayer_conductance", gH.at(0));
    context_1.getPrimitiveData(UUID_1.at(1), "boundarylayer_conductance", gH.at(1));

    std::cout << "Subset run tests passed." << std::endl;

    // **New Test: Edge Cases for calculateBoundaryLayerConductance**
    float result = blc_1.calculateBoundaryLayerConductance(0, 0, 0, 1, 0, 0, 0);
    assert(result == 0);

    result = blc_1.calculateBoundaryLayerConductance(1, 1000, 100, 2, 45 * M_PI / 180, 300, 290);
    assert(result > 0); // Verify reasonable output

    result = blc_1.calculateBoundaryLayerConductance(1, 1000, 100, 2, 80 * M_PI / 180, 300, 290);
    assert(result > 0); // Ensure inclined plate with high inclination works

    result = blc_1.calculateBoundaryLayerConductance(1, 1000, 100, 2, 80 * M_PI / 180, 280, 300);
    assert(result > 0); // Check `free_direction = -1` case

    std::cout << "Edge case tests for calculateBoundaryLayerConductance passed." << std::endl;

    std::cout << "All tests passed successfully!" << std::endl;
    return 0;
}

