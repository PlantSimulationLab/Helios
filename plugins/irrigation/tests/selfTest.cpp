#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"
#include "IrrigationModel.h"
using namespace helios;

const float err_tol = 1e-3f; // Error tolerance for floating-point comparisons

DOCTEST_TEST_CASE("System Creation and Parameter Validation") {
    IrrigationModel model;

    SUBCASE("Empty System") {
        auto summary = model.getSystemSummary();
        DOCTEST_CHECK(summary.find("Total nodes: 0") != std::string::npos);
        DOCTEST_CHECK(summary.find("Total links: 0") != std::string::npos);
    }

    SUBCASE("Complete System Creation") {
        DOCTEST_CHECK_NOTHROW(model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE));

        auto summary = model.getSystemSummary();
        DOCTEST_CHECK(summary.find("waterSource") != std::string::npos);
        DOCTEST_CHECK(summary.find("lateral") != std::string::npos);
    }


    SUBCASE("Valid Parameters") {
        DOCTEST_CHECK_NOTHROW(model.createCompleteSystem(25, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE));
    }

    SUBCASE("Invalid Field Dimensions") {
        DOCTEST_CHECK_THROWS(model.createCompleteSystem(25, 0.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE));
        DOCTEST_CHECK_THROWS(model.createCompleteSystem(25, 100.0, -5.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE));
    }

    SUBCASE("Invalid Spacing Values") {
        DOCTEST_CHECK_THROWS(model.createCompleteSystem(25, 100.0, 50.0, 0.0, 5.0, "vertical", SubmainPosition::MIDDLE));
        DOCTEST_CHECK_THROWS(model.createCompleteSystem(25, 100.0, 50.0, 10.0, -2.0, "vertical", SubmainPosition::MIDDLE));
    }

    SUBCASE("Spacing Exceeds Field Dimensions") {
        DOCTEST_CHECK_THROWS(model.createCompleteSystem(25, 100.0, 50.0, 150.0, 5.0, "vertical", SubmainPosition::MIDDLE));
        DOCTEST_CHECK_THROWS(model.createCompleteSystem(25, 100.0, 50.0, 10.0, 60.0, "vertical", SubmainPosition::MIDDLE));
    }
}

// private function
// DOCTEST_TEST_CASE("Water Source Position Calculation") {
//     IrrigationModel model;
//
//     SUBCASE("Vertical Laterals") {
//         auto pos = model.calculateWaterSourcePosition(100.0, 50.0, "vertical");
//         DOCTEST_CHECK(pos.x == doctest::Approx(100.0/3.0 + 5.0));
//         DOCTEST_CHECK(pos.y == doctest::Approx(50.0 - 50.0/3.0 - 0.5));
//     }
//
//     SUBCASE("Horizontal Laterals") {
//         auto pos = model.calculateWaterSourcePosition(100.0, 50.0, "horizontal");
//         DOCTEST_CHECK(pos.x == doctest::Approx(50.0));
//         DOCTEST_CHECK(pos.y == doctest::Approx(0.0));
//     }
// }


DOCTEST_TEST_CASE("Hydraulic Calculations") {
    IrrigationModel model;

    SUBCASE("Empty System Hydraulics") {
        // Test with no nodes (should handle gracefully)
        auto results = model.calculateHydraulics("PC", 10.0, 25.0);

        DOCTEST_CHECK(results.converged == false);
        DOCTEST_CHECK(results.iterations == 0);
        DOCTEST_CHECK(results.nodalPressures.empty());
        DOCTEST_CHECK(results.flowRates.empty());
    }

    SUBCASE("Small Complete System with PC Nozzle") {
        // Create a small complete system
        model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);

        auto results = model.calculateHydraulics("PC", 15.0, 25.0);

        // Basic validation
        DOCTEST_CHECK(results.converged == true);
        DOCTEST_CHECK(results.iterations > 0);
        DOCTEST_CHECK_FALSE(results.nodalPressures.empty());
        DOCTEST_CHECK_FALSE(results.flowRates.empty());

        // Water source pressure should be maintained
        bool has_water_source_pressure = false;
        for (const auto& pressure : results.nodalPressures) {
            if (pressure == doctest::Approx(25.0).epsilon(0.1)) {
                has_water_source_pressure = true;
                break;
            }
        }
        DOCTEST_CHECK(has_water_source_pressure == true);

        // All pressures should be reasonable
        for (const auto& pressure : results.nodalPressures) {
            DOCTEST_CHECK(pressure >= 0.0);
            DOCTEST_CHECK(pressure <= 30.0); // Should be less than source pressure + margin
        }
    }

    SUBCASE("Small Complete System with NPC Nozzle") {
        model.createCompleteSystem(30.0, 60.0, 40.0, 12.0, 6.0, "vertical", SubmainPosition::SOUTH);

        auto results = model.calculateHydraulics("NPC", 20.0, 30.0);

        DOCTEST_CHECK(results.converged == true);
        DOCTEST_CHECK(results.iterations > 0);

        // NPC should have non-zero flows
        bool has_non_zero_flow = false;
        for (const auto& flow : results.flowRates) {
            if (flow > 0.0) {
                has_non_zero_flow = true;
                break;
            }
        }
        DOCTEST_CHECK(has_non_zero_flow == true);

        // Check pressure gradient (should decrease from source)
        bool pressure_decreases = false;
        for (size_t i = 1; i < results.nodalPressures.size(); ++i) {
            if (results.nodalPressures[i] < results.nodalPressures[0]) {
                pressure_decreases = true;
                break;
            }
        }
        DOCTEST_CHECK(pressure_decreases == true);
    }

    SUBCASE("Different Connection Types") {
        SUBCASE("Vertical Connection") {
            model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
            auto results = model.calculateHydraulics("PC", 15.0, 25.0);
            DOCTEST_CHECK(results.converged == true);
        }

        SUBCASE("Horizontal Connection") {
            model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "horizontal", SubmainPosition::NORTH);
            auto results = model.calculateHydraulics("PC", 15.0, 25.0);
            DOCTEST_CHECK(results.converged == true);
        }
    }

    SUBCASE("Different Submain Positions") {
        SUBCASE("North Submain") {
            model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
            auto results = model.calculateHydraulics("PC", 15.0, 25.0);
            DOCTEST_CHECK(results.converged == true);
        }

        SUBCASE("Middle Submain") {
            model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);
            auto results = model.calculateHydraulics("PC", 15.0, 25.0);
            DOCTEST_CHECK(results.converged == true);
        }

        SUBCASE("South Submain") {
            model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::SOUTH);
            auto results = model.calculateHydraulics("PC", 15.0, 25.0);
            DOCTEST_CHECK(results.converged == true);
        }
    }

    SUBCASE("Zero Pressure Input") {
        model.createCompleteSystem(0.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);

        auto results = model.calculateHydraulics("PC", 10.0, 0.0);

        // With zero pressure, all flows should be zero or very small
        for (const auto& flow : results.flowRates) {
            DOCTEST_CHECK(flow >= 0.0);
            DOCTEST_CHECK(flow <= 1e-6); // Essentially zero
        }
    }

    SUBCASE("High Pressure System") {
        model.createCompleteSystem(50.0, 100.0, 80.0, 15.0, 8.0, "vertical", SubmainPosition::MIDDLE);

        auto results = model.calculateHydraulics("PC", 25.0, 50.0);

        DOCTEST_CHECK(results.converged == true);

        // High pressure should result in higher flows
        double total_flow = 0.0;
        for (const auto& flow : results.flowRates) {
            total_flow += flow;
        }
        DOCTEST_CHECK(total_flow > 0.0);
    }

    SUBCASE("Convergence Validation") {
        model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);

        auto results = model.calculateHydraulics("PC", 10.0, 30.0);

        // Should converge within reasonable iterations
        DOCTEST_CHECK(results.converged == true);
        DOCTEST_CHECK(results.iterations <= 100); // Should not take excessive iterations

        // Converged results should have finite values
        for (const auto& pressure : results.nodalPressures) {
            DOCTEST_CHECK(std::isfinite(pressure));
        }
        for (const auto& flow : results.flowRates) {
            DOCTEST_CHECK(std::isfinite(flow));
        }
    }

    SUBCASE("Invalid Nozzle Type") {
        model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);

        // Should throw for invalid nozzle type
        DOCTEST_CHECK_THROWS_AS(model.calculateHydraulics("INVALID", 10.0, 25.0), std::invalid_argument);
    }

    SUBCASE("Mass Conservation Check") {
        model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);

        auto results = model.calculateHydraulics("PC", 10.0, 30.0);

        // Basic sanity check: total flow should be positive
        double total_flow = 0.0;
        for (const auto& flow : results.flowRates) {
            total_flow += flow;
        }

        DOCTEST_CHECK(total_flow > 0.0);
        DOCTEST_CHECK(total_flow < 1000.0); // Reasonable upper bound
    }

    SUBCASE("Pressure Gradient Validation") {
        model.createCompleteSystem(40.0, 80.0, 60.0, 12.0, 6.0, "vertical", SubmainPosition::NORTH);

        auto results = model.calculateHydraulics("PC", 15.0, 40.0);

        // Pressures should generally decrease away from water source
        // (This is a simplified check - actual hydraulic gradient may vary)
        bool pressure_variation_exists = false;
        double max_pressure = *std::max_element(results.nodalPressures.begin(), results.nodalPressures.end());
        double min_pressure = *std::min_element(results.nodalPressures.begin(), results.nodalPressures.end());

        if (max_pressure - min_pressure > 1.0) {
            pressure_variation_exists = true;
        }

        DOCTEST_CHECK(pressure_variation_exists == true);
    }

    SUBCASE("Mass Conservation Check") {
        model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);

        auto results = model.calculateHydraulics("PC", 10.0, 25.0);

        // Basic mass conservation: sum of emitter flows should be reasonable
        double total_emitter_flow = 0.0;
        for (const auto& flow : results.flowRates) {
            total_emitter_flow += flow;
        }

        DOCTEST_CHECK(total_emitter_flow > 0.0);
        DOCTEST_CHECK(total_emitter_flow < 100.0); // Reasonable upper bound
    }

}



int IrrigationModel::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}


